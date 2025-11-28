"""Conv1D + LSTM model for sequence prediction."""

import torch
import torch.nn as nn
from typing import List


class ConvLSTMModel(nn.Module):
    """Conv1D + LSTM model with multi-horizon outputs."""
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        horizons: List[int] = [1, 5, 20, 120],
        conv_filters: List[int] = [64, 64],
        conv_kernel_sizes: List[int] = [3, 3],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        """
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            horizons: List of prediction horizons
            conv_filters: Number of filters for each Conv1D layer
            conv_kernel_sizes: Kernel sizes for each Conv1D layer
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            use_uncertainty: If True, output mean and variance for each horizon
        """
        super(ConvLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.use_uncertainty = use_uncertainty
        
        # Conv1D layers (causal padding)
        conv_layers = []
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(conv_filters, conv_kernel_sizes):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1),  # Causal padding
                    padding_mode='zeros'
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output heads (one per horizon)
        self.output_heads = nn.ModuleDict()
        
        for horizon in horizons:
            if use_uncertainty:
                # Mean and log-variance outputs
                self.output_heads[f'h{horizon}_mean'] = nn.Sequential(
                    nn.Linear(lstm_hidden, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
                self.output_heads[f'h{horizon}_logvar'] = nn.Sequential(
                    nn.Linear(lstm_hidden, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )
            else:
                # Single output
                self.output_heads[f'h{horizon}'] = nn.Sequential(
                    nn.Linear(lstm_hidden, 64),
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
            Dict mapping horizon -> predictions (or (mean, logvar) if uncertainty)
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1D: (batch, features, time)
        x = x.transpose(1, 2)
        
        # Apply Conv1D layers
        x = self.conv_layers(x)
        
        # Transpose back: (batch, time, features)
        x = x.transpose(1, 2)
        
        # Take only valid (non-padded) part for causal convolution
        x = x[:, :self.sequence_length, :]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden)
        
        # Output heads
        outputs = {}
        
        for horizon in self.horizons:
            if self.use_uncertainty:
                mean = self.output_heads[f'h{horizon}_mean'](last_hidden)
                logvar = self.output_heads[f'h{horizon}_logvar'](last_hidden)
                outputs[horizon] = (mean.squeeze(-1), logvar.squeeze(-1))
            else:
                pred = self.output_heads[f'h{horizon}'](last_hidden)
                outputs[horizon] = pred.squeeze(-1)
        
        return outputs
    
    def predict_mc(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Dict[int, tuple]:
        """
        Monte Carlo dropout prediction for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
        
        Returns:
            Dict mapping horizon -> (mean, std)
        """
        self.train()  # Keep dropout active
        
        predictions = {h: [] for h in self.horizons}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x)
                for horizon, pred in outputs.items():
                    predictions[horizon].append(pred.cpu().numpy())
        
        # Compute mean and std
        results = {}
        for horizon in self.horizons:
            pred_array = np.array(predictions[horizon])
            mean = np.mean(pred_array, axis=0)
            std = np.std(pred_array, axis=0)
            results[horizon] = (mean, std)
        
        return results

