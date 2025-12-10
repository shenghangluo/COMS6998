"""Main confidence prediction model integrating backbone LLM and MLP head."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

from .mlp_head import ConfidenceMLP


class ConfidenceModel(nn.Module):
    """
    Confidence prediction model combining a pretrained LLM backbone with an MLP head.

    The model:
    1. Takes text input with <CONFIDENCE> tokens inserted after reasoning steps
    2. Processes through the LLM backbone to get embeddings
    3. Extracts embeddings at <CONFIDENCE> token positions
    4. Passes these embeddings through the MLP head to predict confidence scores

    Args:
        model_name: HuggingFace model name/path
        mlp_config: Configuration dict for the MLP head
        device_map: Device mapping for model (default: "auto")
        torch_dtype: Data type for model weights (default: torch.bfloat16)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        mlp_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Load backbone LLM
        print(f"Loading backbone model: {model_name}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Get model dimension
        self.hidden_size = self.backbone.config.hidden_size

        # Initialize MLP head
        if mlp_config is None:
            mlp_config = {
                "input_dim": self.hidden_size,
                "hidden_dims": [1024, 256],
                "dropout": 0.1,
                "activation": "gelu"
            }
        else:
            # Update input_dim if not specified
            if mlp_config.get("input_dim") is None:
                mlp_config["input_dim"] = self.hidden_size

        self.mlp_head = ConfidenceMLP(**mlp_config)

        # Move MLP head to same device and dtype as backbone
        if hasattr(self.backbone, 'device'):
            self.mlp_head = self.mlp_head.to(self.backbone.device)
        # Convert all parameters to the correct dtype
        self.mlp_head = self.mlp_head.to(dtype=torch_dtype)

        self.confidence_token_id = None  # Will be set after tokenizer adds token

    def set_confidence_token_id(self, token_id: int):
        """Set the ID of the <CONFIDENCE> token."""
        self.confidence_token_id = token_id
        print(f"Confidence token ID set to: {token_id}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        confidence_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            confidence_positions: Positions of <CONFIDENCE> tokens, shape (batch_size, num_steps)
                                If None, automatically detect from input_ids
            labels: Ground truth confidence scores of shape (batch_size, num_steps)
            return_dict: Whether to return a dict or tuple

        Returns:
            Dict containing:
                - predictions: Predicted confidence scores
                - loss: MSE loss if labels provided
                - hidden_states: Extracted confidence token embeddings (optional)
        """
        # Get outputs from backbone
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,  # Disable KV cache to save memory during training
            return_dict=True
        )

        # Get last hidden states (batch_size, seq_len, hidden_size)
        hidden_states = backbone_outputs.hidden_states[-1]

        # Extract embeddings at confidence token positions
        if confidence_positions is None:
            # Automatically find confidence token positions
            confidence_positions = self._find_confidence_positions(input_ids)

        confidence_embeddings = self._extract_embeddings_at_positions(
            hidden_states, confidence_positions
        )

        # Pass through MLP head to get predictions
        predictions = self.mlp_head(confidence_embeddings)  # (batch_size, num_steps, 1)
        predictions = predictions.squeeze(-1)  # (batch_size, num_steps)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Validate labels are in expected [0, 1] range
            valid_mask = labels >= 0
            if valid_mask.any():
                valid_labels = labels[valid_mask]

                # Check for outliers and clamp
                if (valid_labels < 0).any() or (valid_labels > 1).any():
                    outlier_count = ((valid_labels < 0) | (valid_labels > 1)).sum().item()
                    print(f"WARNING: {outlier_count} label values out of [0, 1] range detected!")
                    print(f"  Min: {valid_labels.min().item():.4f}, Max: {valid_labels.max().item():.4f}")
                    valid_labels = torch.clamp(valid_labels, 0.0, 1.0)

                # MSE loss for regression
                loss_fn = nn.MSELoss()
                loss = loss_fn(predictions[valid_mask], valid_labels)

        if return_dict:
            return {
                "predictions": predictions,
                "loss": loss,
                "confidence_embeddings": confidence_embeddings
            }
        else:
            return (predictions, loss, confidence_embeddings)

    def _find_confidence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Find positions of <CONFIDENCE> tokens in input_ids.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, max_num_steps) with positions.
            Padded with -1 for missing positions.
        """
        if self.confidence_token_id is None:
            raise ValueError("Confidence token ID not set. Call set_confidence_token_id() first.")

        batch_size, seq_len = input_ids.shape
        positions_list = []

        for i in range(batch_size):
            # Find all positions where confidence token appears
            positions = (input_ids[i] == self.confidence_token_id).nonzero(as_tuple=True)[0]
            positions_list.append(positions)

        # Pad to same length
        max_steps = max(len(p) for p in positions_list) if positions_list else 0
        if max_steps == 0:
            return torch.full((batch_size, 1), -1, dtype=torch.long, device=input_ids.device)

        padded_positions = torch.full(
            (batch_size, max_steps), -1, dtype=torch.long, device=input_ids.device
        )

        for i, positions in enumerate(positions_list):
            if len(positions) > 0:
                padded_positions[i, :len(positions)] = positions

        return padded_positions

    def _extract_embeddings_at_positions(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings at specified positions.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            positions: Positions to extract from, shape (batch_size, num_steps)

        Returns:
            Extracted embeddings of shape (batch_size, num_steps, hidden_size)
        """
        batch_size, num_steps = positions.shape
        hidden_size = hidden_states.size(-1)

        embeddings = torch.zeros(
            batch_size, num_steps, hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for i in range(batch_size):
            for j in range(num_steps):
                pos = positions[i, j]
                if pos >= 0:  # Valid position
                    embeddings[i, j] = hidden_states[i, pos]

        return embeddings

    def generate_with_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **generation_kwargs
    ):
        """
        Generate text and extract confidence scores during generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            Dict with generated_ids and confidence_scores
        """
        # Generate text
        outputs = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **generation_kwargs
        )

        generated_ids = outputs.sequences

        # Extract confidence scores from generated sequence
        with torch.no_grad():
            forward_outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=None,
                return_dict=True
            )

        return {
            "generated_ids": generated_ids,
            "confidence_scores": forward_outputs["predictions"]
        }


if __name__ == "__main__":
    # Test model initialization
    print("Testing ConfidenceModel...")

    model = ConfidenceModel(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        mlp_config={
            "input_dim": 4096,
            "hidden_dims": [1024, 256],
            "dropout": 0.1
        }
    )

    print(f"\nModel initialized successfully!")
    print(f"Backbone hidden size: {model.hidden_size}")
    print(f"MLP head: {model.mlp_head}")
