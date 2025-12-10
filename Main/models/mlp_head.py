"""Standalone MLP head for confidence prediction ([0, 1] regression)."""

import torch
import torch.nn as nn
from typing import List


class ConfidenceMLP(nn.Module):
    """
    Multi-layer perceptron for predicting confidence scores.

    Takes the embedding of the <CONFIDENCE> token and predicts a confidence
    score in the range [0, 1] via regression with sigmoid output activation.

    Args:
        input_dim: Dimension of the input embedding (default: 4096 for Qwen3-4B)
        hidden_dims: List of hidden layer dimensions (default: [1024, 256])
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: GELU)
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 256]

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (regression to [0, 1])
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1] range

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation.lower(), nn.GELU())

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, confidence_token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            confidence_token_embeddings: Tensor of shape (batch_size, input_dim)
                or (batch_size, num_steps, input_dim)

        Returns:
            Confidence scores of shape (batch_size, 1) or (batch_size, num_steps, 1)
        """
        return self.mlp(confidence_token_embeddings)


if __name__ == "__main__":
    # Test the MLP head
    batch_size = 4
    input_dim = 4096

    mlp = ConfidenceMLP(input_dim=input_dim)
    print(f"MLP Architecture:\n{mlp}\n")

    # Test forward pass
    dummy_embeddings = torch.randn(batch_size, input_dim)
    output = mlp(dummy_embeddings)

    print(f"Input shape: {dummy_embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output values: {output.squeeze()}")

    # Test with multiple steps per example
    num_steps = 5
    dummy_multi_step = torch.randn(batch_size, num_steps, input_dim)
    output_multi = mlp(dummy_multi_step)
    print(f"\nMulti-step input shape: {dummy_multi_step.shape}")
    print(f"Multi-step output shape: {output_multi.shape}")
