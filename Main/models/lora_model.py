"""LoRA-based confidence prediction model for parameter-efficient fine-tuning."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any

from .mlp_head import ConfidenceMLP


class ConfidenceModelLoRA(nn.Module):
    """
    LoRA-based confidence prediction model.

    Instead of full fine-tuning, this model:
    1. Freezes the base LLM weights
    2. Adds LoRA adapters to attention layers
    3. Trains only the LoRA parameters + MLP head

    This significantly reduces memory and computational requirements while
    maintaining competitive performance.

    Args:
        model_name: HuggingFace model name/path
        lora_config: Configuration dict for LoRA (r, alpha, dropout, target_modules)
        mlp_config: Configuration dict for the MLP head
        device_map: Device mapping for model (default: "auto")
        torch_dtype: Data type for model weights (default: torch.bfloat16)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        lora_config: Optional[Dict[str, Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Load base model
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Get model dimension
        self.hidden_size = base_model.config.hidden_size

        # Setup LoRA configuration
        if lora_config is None:
            lora_config = {
                "r": 16,  # LoRA rank
                "lora_alpha": 32,  # LoRA alpha scaling
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }

        # Convert to LoraConfig object
        peft_config = LoraConfig(**lora_config)

        # Apply LoRA to base model
        print(f"Applying LoRA with r={lora_config.get('r', 16)}, alpha={lora_config.get('lora_alpha', 32)}")
        self.backbone = get_peft_model(base_model, peft_config)

        # Print trainable parameters
        self.backbone.print_trainable_parameters()

        # Initialize MLP head (same as full fine-tuning model)
        if mlp_config is None:
            mlp_config = {
                "input_dim": self.hidden_size,
                "hidden_dims": [1024, 256],
                "dropout": 0.1,
                "activation": "gelu"
            }
        else:
            # Ensure input_dim is set to model's hidden size if not specified
            if "input_dim" not in mlp_config or mlp_config["input_dim"] is None:
                mlp_config["input_dim"] = self.hidden_size

        self.mlp_head = ConfidenceMLP(**mlp_config)

        # Move MLP head to same device and dtype as backbone
        if hasattr(self.backbone, 'device'):
            self.mlp_head = self.mlp_head.to(device=self.backbone.device)
        # Convert all parameters to the correct dtype
        self.mlp_head = self.mlp_head.to(dtype=torch_dtype)

        self.confidence_token_id = None

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
            labels: Ground truth confidence scores of shape (batch_size, num_steps)
            return_dict: Whether to return a dict or tuple

        Returns:
            Dict containing predictions, loss, and embeddings
        """
        # Get outputs from LoRA-enhanced backbone
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get last hidden states
        hidden_states = backbone_outputs.hidden_states[-1]

        # Extract embeddings at confidence token positions
        if confidence_positions is None:
            confidence_positions = self._find_confidence_positions(input_ids)

        confidence_embeddings = self._extract_embeddings_at_positions(
            hidden_states, confidence_positions
        )

        # Pass through MLP head
        predictions = self.mlp_head(confidence_embeddings)
        predictions = predictions.squeeze(-1)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            valid_mask = labels >= 0
            if valid_mask.any():
                loss = loss_fn(predictions[valid_mask], labels[valid_mask])

        if return_dict:
            return {
                "predictions": predictions,
                "loss": loss,
                "confidence_embeddings": confidence_embeddings
            }
        else:
            return (predictions, loss, confidence_embeddings)

    def _find_confidence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find positions of <CONFIDENCE> tokens in input_ids."""
        if self.confidence_token_id is None:
            raise ValueError("Confidence token ID not set. Call set_confidence_token_id() first.")

        batch_size, seq_len = input_ids.shape
        positions_list = []

        for i in range(batch_size):
            positions = (input_ids[i] == self.confidence_token_id).nonzero(as_tuple=True)[0]
            positions_list.append(positions)

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
        """Extract embeddings at specified positions."""
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
                if pos >= 0:
                    embeddings[i, j] = hidden_states[i, pos]

        return embeddings

    def save_lora_adapter(self, save_path: str):
        """Save only the LoRA adapter weights."""
        self.backbone.save_pretrained(save_path)
        print(f"LoRA adapter saved to {save_path}")

    def save_mlp_head(self, save_path: str):
        """Save MLP head weights."""
        torch.save(self.mlp_head.state_dict(), save_path)
        print(f"MLP head saved to {save_path}")

    def load_lora_adapter(self, load_path: str):
        """Load LoRA adapter weights."""
        # This is handled by PEFT library
        print(f"Loading LoRA adapter from {load_path}")

    def load_mlp_head(self, load_path: str):
        """Load MLP head weights."""
        self.mlp_head.load_state_dict(torch.load(load_path))
        print(f"MLP head loaded from {load_path}")

    def merge_and_unload(self):
        """Merge LoRA weights into base model and return unloaded model."""
        return self.backbone.merge_and_unload()


if __name__ == "__main__":
    # Test LoRA model initialization
    print("Testing ConfidenceModelLoRA...")

    model = ConfidenceModelLoRA(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
        mlp_config={
            "input_dim": 4096,
            "hidden_dims": [1024, 256],
            "dropout": 0.1
        }
    )

    print(f"\nLoRA model initialized successfully!")
    print(f"Backbone hidden size: {model.hidden_size}")
