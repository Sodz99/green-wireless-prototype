import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class RFNpyDataset(Dataset):
    """Dataset for RF modulation classification from .npy files"""

    def __init__(self, x_path, y_path):
        self.x_data = np.load(x_path).astype(np.float32)  # (N, 2, 2048)
        self.y_data = np.load(y_path).astype(np.int64)    # (N,)

        print(f"Loaded dataset: {self.x_data.shape}, labels: {self.y_data.shape}")
        print(f"Classes: {np.unique(self.y_data)}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x_data[idx])  # Shape: (2, 2048)
        y = torch.tensor(self.y_data[idx], dtype=torch.long)
        return x, y


def get_dataloaders(config, data_dir='data'):
    """Create train/val/test dataloaders"""

    # Create datasets
    train_dataset = RFNpyDataset(
        os.path.join(data_dir, 'train_x.npy'),
        os.path.join(data_dir, 'train_y.npy')
    )

    val_dataset = RFNpyDataset(
        os.path.join(data_dir, 'val_x.npy'),
        os.path.join(data_dir, 'val_y.npy')
    )

    test_dataset = RFNpyDataset(
        os.path.join(data_dir, 'test_x.npy'),
        os.path.join(data_dir, 'test_y.npy')
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    import yaml
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Test one batch
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
    print(f"Data types: {x.dtype}, {y.dtype}")