"""PyTorch datasets for sequence models."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class SequenceDataset(Dataset):
    """Dataset for sequence models with multi-horizon labels."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: Dict[int, np.ndarray],
        asset_ids: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None
    ):
        """
        Args:
            sequences: Array of shape (N, L, D) where N=samples, L=sequence_length, D=features
            labels: Dict mapping horizon -> array of shape (N,)
            asset_ids: Optional array of asset_ids for each sample
            dates: Optional array of dates for each sample
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = {h: torch.FloatTensor(l) for h, l in labels.items()}
        self.asset_ids = asset_ids
        self.dates = dates
        self.num_samples = len(sequences)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = {
            'sequence': self.sequences[idx],
            'labels': {h: l[idx] for h, l in self.labels.items()}
        }
        
        if self.asset_ids is not None:
            sample['asset_id'] = self.asset_ids[idx]
        if self.dates is not None:
            sample['date'] = self.dates[idx]
        
        return sample

