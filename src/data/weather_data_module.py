import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class WeatherDataset(Dataset):
    def __init__(self, data_dir: str, location: str):
        self.data = pd.read_csv(os.path.join(data_dir, f"{location}_2018-01-01_2023-12-31_aligned.csv"))
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        
        # Prepare features and targets using the correct column names
        feature_columns = [
            'temperature',          # Open-Meteo temperature
            'humidity',            # Open-Meteo humidity
            'wind_speed_model',    # Open-Meteo wind speed
            'wind_direction_model', # Open-Meteo wind direction
            'cloud_cover_low',     # Low cloud cover
            'cloud_cover_mid',     # Mid cloud cover
            'cloud_cover_high'     # High cloud cover
        ]
        
        # Handle missing values
        self.data = self.data.ffill().bfill()
        
        # Normalize features
        self.features = self.data[feature_columns].values
        self.feature_means = np.nanmean(self.features, axis=0)
        self.feature_stds = np.nanstd(self.features, axis=0)
        self.features = (self.features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Normalize targets
        self.targets = self.data['temp_avg'].values
        self.target_mean = np.nanmean(self.targets)
        self.target_std = np.nanstd(self.targets)
        self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return data as a dictionary with 'input' and 'target' keys
        # Add sequence dimension (1) to make it 3D: (batch_size, sequence_length, features)
        return {
            'input': torch.FloatTensor(self.features[idx]).unsqueeze(0),  # Add sequence dimension
            'target': torch.FloatTensor([self.targets[idx]]).unsqueeze(0)  # Add sequence dimension to match model output
        }
    
    def denormalize_target(self, normalized_target):
        """Convert normalized target back to original scale."""
        return normalized_target * self.target_std + self.target_mean

class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define locations
        self.locations = ['London', 'Amsterdam', 'Paris']
        
    def setup(self, stage=None):
        # Create datasets for each location
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
        for location in self.locations:
            dataset = WeatherDataset(self.data_dir, location)
            
            # Split data: 70% train, 15% val, 15% test
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, len(dataset) - train_size - val_size]
            )
            
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
        
        # Combine datasets from all locations
        self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
        self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)
        self.test_dataset = torch.utils.data.ConcatDataset(self.test_datasets)
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        ) 