import torch
import torch.nn as nn
import numpy as np
import logging
import os
from pathlib import Path
import pandas as pd
import json




class PropertyScaler(nn.Module):
    """
    Automatically scales molecular properties to balance their contribution to the model.
    
    This tracks statistics during training and normalizes properties to have similar distributions.
    """
    
    def __init__(self, property_name, scale_file="property_scales.json", initial_mean= None , initial_std= None):
        super().__init__()
        self.property_name = property_name
        self.scale_file = scale_file
        self.DEFAULT_STATS ={
              "logp": {"mean": 3.93964146960539,"std": 2.1115346757029},
            "tpsa": {"mean": 58.18354678874095,"std": 38.648784762734},
            'molwt': {'mean': 350.0, 'std': 150.0},
            'qed': {'mean': 0.5, 'std': 0.2},
            'sas': {'mean': 3.0292936161687023, 'std': 0.736605966214189},
            'homo': {'mean': -5.641539097413394, 'std': 0.6242975864588},
            'lumo': {'mean': -1.652423883026786, 'std': 0.8706566300680232},
            }
        
        # Use property-specific defaults if available, otherwise use provided values or general defaults
        if initial_mean is None:
            initial_mean = self.DEFAULT_STATS.get(property_name, {}).get('mean', 0.0)
        if initial_std is None:
            initial_std = self.DEFAULT_STATS.get(property_name, {}).get('std', 1.0)
        
        # Learnable parameters to store mean and std
        self.register_buffer('mean', torch.tensor(initial_mean, dtype=torch.float))
        self.register_buffer('std', torch.tensor(initial_std, dtype=torch.float))
        
        # Training mode statistics
        self.tracking = False
        self.sum = 0
        self.sum_squared = 0
        self.count = 0
        
        # Try to load existing statistics
        # Try to load existing statistics or create new file with defaults
        if not self.load_stats():
            self.save_stats()  # Create the file with initial values
        
        
    
    def load_stats(self):
        """Load statistics from file if available"""
        if os.path.exists(self.scale_file):
            try:
                with open(self.scale_file, 'r') as f:
                    stats = json.load(f)
                
                if self.property_name in stats:
                    prop_stats = stats[self.property_name]
                    self.mean = torch.tensor(prop_stats['mean'], dtype=torch.float)
                    self.std = torch.tensor(prop_stats['std'], dtype=torch.float)
                    logging.info(f"Loaded statistics for {self.property_name}: mean={self.mean.item():.4f}, std={self.std.item():.4f}")
                    return True
            except Exception as e:
                logging.warning(f"Failed to load statistics for {self.property_name}: {e}")
        else:
            logging.info(f"Statistics file {self.scale_file} not found. Creating new file.")
        
        return False
    def save_stats(self):
        """Save current statistics to file"""
        # Create directory if needed
        Path(os.path.dirname(self.scale_file)).mkdir(parents=True, exist_ok=True)
        
        # Load existing stats if available
        stats = {}
        if os.path.exists(self.scale_file):
            try:
                with open(self.scale_file, 'r') as f:
                    stats = json.load(f)
            except:
                pass
        
        # Update with current stats
        stats[self.property_name] = {
            'mean': float(self.mean.item()),
            'std': float(self.std.item())
        }
        
        # Save to file
        with open(self.scale_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Saved statistics for {self.property_name}: mean={self.mean.item():.4f}, std={self.std.item():.4f}")
        
    def track_statistics(self, enable=True):
        """Enable/disable statistic tracking"""
        self.tracking = enable
        if enable:
            self.sum = 0
            self.sum_squared = 0
            self.count = 0
    
    def update_statistics(self):
        """Update mean and std based on tracked statistics"""
        if self.count > 0:
            # Calculate statistics
            new_mean = self.sum / self.count
            # Avoid variance to be negative
            variance = max(0.0, (self.sum_squared / self.count) - (new_mean * new_mean))
            new_std = torch.sqrt(variance)
            
            # Avoid division by zero
            if new_std < 1e-8:
                new_std = torch.tensor(1.0, device=new_std.device)
            
            # Update parameters
            self.mean = new_mean
            self.std = new_std
            
            # Save to file
            self.save_stats()
            
            # Reset tracking
            self.sum = 0
            self.sum_squared = 0
            self.count = 0
            
    def forward(self, x, normalize=True):
        """
        Normalize or denormalize property values.
        
        Args:
            x: Property tensor
            normalize: If True, normalize (for inputs); if False, denormalize (for outputs)
        
        Returns:
            Normalized or denormalized tensor
        """
        # Skip NaN values
        mask = ~torch.isnan(x)
        if not mask.any():
            return x
        
        # Update statistics if tracking
        if self.tracking and normalize:
            valid_x = x[mask]
            self.sum += torch.sum(valid_x)
            self.sum_squared += torch.sum(valid_x * valid_x)
            self.count += valid_x.numel()
        
        # Apply normalization/denormalization
        result = torch.clone(x)
        if normalize:
            # (x - mean) / std
            result[mask] = (x[mask] - self.mean) / self.std
        else:
            # x * std + mean
            result[mask] = x[mask] * self.std + self.mean
        
        return result