"""Temporal Convolutional Network (TCN) model."""

import torch
import torch.nn as nn
from typing import List


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence prediction."""
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        horizons: List[int] = [1, 5, 20, 120],
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        """
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            horizons: List of prediction horizons
            num_channels: Number of channels in each TCN block
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            use_uncertainty: If True, output mean and variance
        """
        super(TCNModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.use_uncertainty = use_uncertainty
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # Output heads
        self.output_heads = nn.ModuleDict()
        final_channels = num_channels[-1]
        
        for horizon in horizons:
            if use_uncertainty:
                self.output_heads[f'h{horizon}_mean'] = nn.Sequential(
                    nn.Linear(final_channels, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
                self.output_heads[f'h{horizon}_logvar'] = nn.Sequential(
                    nn.Linear(final_channels, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            else:
                self.output_heads[f'h{horizon}'] = nn.Sequential(
                    nn.Linear(final_channels, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
        
        Returns:
            Dict mapping horizon -> predictions
        """
        # Transpose for Conv1D: (batch, features, time)
        x = x.transpose(1, 2)
        
        # TCN layers
        x = self.tcn(x)
        
        # Global average pooling over time dimension
        x = x.mean(dim=2)  # (batch, channels)
        
        # Output heads
        outputs = {}
        
        for horizon in self.horizons:
            if self.use_uncertainty:
                mean = self.output_heads[f'h{horizon}_mean'](x)
                logvar = self.output_heads[f'h{horizon}_logvar'](x)
                outputs[horizon] = (mean.squeeze(-1), logvar.squeeze(-1))
            else:
                pred = self.output_heads[f'h{horizon}'](x)
                outputs[horizon] = pred.squeeze(-1)
        
        return outputs
    
    def predict_mc(self, x: torch.Tensor, n_samples: int = 50) -> Dict[int, tuple]:
        """Monte Carlo dropout prediction."""
        self.train()
        
        predictions = {h: [] for h in self.horizons}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x)
                for horizon, pred in outputs.items():
                    if isinstance(pred, tuple):
                        predictions[horizon].append(pred[0].cpu().numpy())
                    else:
                        predictions[horizon].append(pred.cpu().numpy())
        
        results = {}
        for horizon in self.horizons:
            pred_array = np.array(predictions[horizon])
            mean = np.mean(pred_array, axis=0)
            std = np.std(pred_array, axis=0)
            results[horizon] = (mean, std)
        
        return results

