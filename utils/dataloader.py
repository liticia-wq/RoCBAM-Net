import torch
from torch.utils.data import DataLoader
from .dataset import BreastMultiPhaseDataset

def get_dataloaders(data_dir, batch_size=8, num_workers=2):
    """
    Constructs PyTorch DataLoaders for Training, Validation, and Testing phases.
    """
    train_dataset = BreastMultiPhaseDataset(base_path=data_dir, mode='train', augment=True)
    val_dataset   = BreastMultiPhaseDataset(base_path=data_dir, mode='val', augment=False)
    test_dataset  = BreastMultiPhaseDataset(base_path=data_dir, mode='test', augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
