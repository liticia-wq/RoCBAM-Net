import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.roc_bam_net import RoCBAMNet
from models.losses import WeightedBCEDiceLoss, get_dice
from utils.dataset import BreastMultiPhaseDataset


def get_model(name, in_channels=9, out_channels=1):
    """
    Model factory function initializing specified architecture.
    """
    if name == 'rocbam':
        return RoCBAMNet(in_channels=in_channels, out_channels=out_channels)
    elif name == 'robunet':
        raise NotImplementedError(
            "Import your RobU-Net baseline class here: "
            "e.g., 'from models.rob_u_net import RobUNet' and return RobUNet()."
        )
    else:
        raise ValueError(f"Unknown model architecture: {name}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    train_dataset = BreastMultiPhaseDataset(args.data_dir, mode='train', augment=True)
    val_dataset = BreastMultiPhaseDataset(args.data_dir, mode='val', augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = get_model(args.model, in_channels=9, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=3
    )
    criterion = WeightedBCEDiceLoss(weight=args.loss_weight)

    best_dice = 0.0
    trigger_times = 0

    print(f"🟢 Training starting | Model: {args.model} | Device: {device} | Noise path: {args.data_dir}")

    for epoch in range(1, args.epochs + 1):
       
        model.train()
        running_train_dice = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_dice += get_dice(outputs, masks).item()

        avg_train_dice = running_train_dice / len(train_loader)

       
        model.eval()
        running_val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                running_val_dice += get_dice(outputs, masks).item()

        avg_val_dice = running_val_dice / len(val_loader)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f}")

     
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            trigger_times = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"⭐ Model checkpoint saved to {args.save_path} (Val Dice: {best_dice:.4f})")
        else:
            trigger_times += 1
            if trigger_times >= args.patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Phase DCE-MRI Tumor Segmentation Training Routine")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to dataset root folder containing train/val splits")
    parser.add_argument('--model', type=str, default='rocbam', choices=['rocbam', 'robunet'],
                        help="Model architecture selector ('rocbam' or 'robunet')")
    parser.add_argument('--epochs', type=int, default=100, help="Maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per iteration")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience threshold")
    parser.add_argument('--loss_weight', type=float, default=50.0, help="Weight parameter for Weighted BCE Loss")
    parser.add_argument('--num_workers', type=int, default=2, help="DataLoader parallel process workers")
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.pth',
                        help="Output path for saving target weights")

    args = parser.parse_args()
    train(args)
