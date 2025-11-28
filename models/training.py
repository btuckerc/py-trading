"""Training loops for sequence models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from models.torch.losses import MultiHorizonLoss
from models.splits import TimeSplit


class SequenceTrainer:
    """Trainer for sequence models."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultiHorizonLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "artifacts/models"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)
            labels = {h: l.to(self.device) for h, l in batch['labels'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(self.device)
                labels = {h: l.to(self.device) for h, l in batch['labels'].items()}
                
                predictions = self.model(sequences)
                loss = self.loss_fn(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """Train model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best:
                        self.save_checkpoint(f"best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.save_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

