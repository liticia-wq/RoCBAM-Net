import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.roc_bam_net import RoCBAMNet
from models.losses import WeightedBCEDiceLoss, get_dice
from utils.dataset import BreastMultiPhaseDataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = BreastMultiPhaseDataset(args.data_dir, mode='train')
    val_dataset = BreastMultiPhaseDataset(args.data_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RoCBAMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedBCEDiceLoss(weight=50.0)

    best_dice = 0.0
    trigger_times = 0

    print("🟢 Starting RoCBAM-Net Training...")

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
        print(f"Epoch {epoch:02d} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            trigger_times = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"⭐ New Best Model Saved (Val Dice: {best_dice:.4f})")
        else:
            trigger_times += 1
            if trigger_times >= args.patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to processed dataset")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='models/checkpoints/best_rocbamnet.pth')
    args = parser.parse_args()
    train(args)
