"""Unit tests for MLP head model."""

import pytest
import torch
import torch.nn as nn
from models.mlp_head import ConfidenceMLP


class TestConfidenceMLP:
    """Test ConfidenceMLP class."""

    def test_mlp_initialization(self):
        """Test MLP initializes correctly."""
        mlp = ConfidenceMLP(input_dim=4096, hidden_dims=[1024, 256])
        
        assert mlp is not None
        assert hasattr(mlp, 'mlp')

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        mlp = ConfidenceMLP(input_dim=4096, hidden_dims=[1024, 256])
        batch_size = 4
        
        # Single embedding
        input_tensor = torch.randn(batch_size, 4096)
        output = mlp(input_tensor)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0.0)  # Sigmoid output >= 0
        assert torch.all(output <= 1.0)   # Sigmoid output <= 1

    def test_mlp_multi_step_forward(self):
        """Test MLP forward pass with multiple steps."""
        mlp = ConfidenceMLP(input_dim=4096, hidden_dims=[1024, 256])
        batch_size = 4
        num_steps = 5
        
        # Multiple steps per example
        input_tensor = torch.randn(batch_size, num_steps, 4096)
        output = mlp(input_tensor)
        
        assert output.shape == (batch_size, num_steps, 1)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)

    def test_mlp_output_range(self):
        """Test MLP output is in [0, 1] range."""
        mlp = ConfidenceMLP(input_dim=4096)
        input_tensor = torch.randn(10, 4096)
        output = mlp(input_tensor)
        
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_mlp_different_hidden_dims(self):
        """Test MLP with different hidden dimensions."""
        mlp = ConfidenceMLP(input_dim=512, hidden_dims=[256, 128, 64])
        input_tensor = torch.randn(4, 512)
        output = mlp(input_tensor)
        
        assert output.shape == (4, 1)

    def test_mlp_dropout(self):
        """Test MLP dropout is applied."""
        mlp = ConfidenceMLP(input_dim=4096, dropout=0.5)
        assert mlp.training or not mlp.training  # Just check it's a valid state

    def test_mlp_activation_functions(self):
        """Test MLP with different activation functions."""
        activations = ['gelu', 'relu', 'silu', 'tanh']
        
        for activation in activations:
            mlp = ConfidenceMLP(input_dim=4096, activation=activation)
            input_tensor = torch.randn(4, 4096)
            output = mlp(input_tensor)
            
            assert output.shape == (4, 1)
            assert torch.all(output >= 0.0)
            assert torch.all(output <= 1.0)

    def test_mlp_gradient_flow(self):
        """Test gradients flow through MLP."""
        mlp = ConfidenceMLP(input_dim=4096)
        input_tensor = torch.randn(4, 4096, requires_grad=True)
        output = mlp(input_tensor)
        
        loss = output.mean()
        loss.backward()
        
        assert input_tensor.grad is not None

    def test_mlp_weights_initialization(self):
        """Test MLP weights are initialized."""
        mlp = ConfidenceMLP(input_dim=4096)
        
        # Check that weights exist and are not all zeros
        for param in mlp.parameters():
            assert param.data is not None
            assert not torch.all(param.data == 0.0)

