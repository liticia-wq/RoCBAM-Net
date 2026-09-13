import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.roc_bam_net import RoCBAMNet
from models.losses import WeightedBCEDiceLoss, get_dice
from utils.dataset import BreastMultiPhaseDataset

# If you want --model to also run your RobU-Net baseline, import it here, e.g.:
# from models.rob_u_net import RobUNet

def get_model(name):
    if name == 'rocbam':
        return RoCBAMNet()
    elif name == 'robunet':
        raise NotImplementedError(
            "Import your RobU-Net class above and return it here, "
            "e.g. 'from models.rob_u_net import RobUNet' then 'return RobUNet()'."
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = BreastMultiPhaseDataset(args.data_dir, mode='train')
    val_dataset = BreastMultiPhaseDataset(args.data_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=3
    )
    criterion = WeightedBCEDiceLoss(weight=50.0)

    best_dice = 0.0
    trigger_times = 0

    print(f"🟢 Starting {args.model} training (noise regime: {args.data_dir})...")

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

        # Step the LR scheduler on validation Dice (matches actual training runs)
        scheduler.step(avg_val_dice)

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
    parser.add_argument('--data_dir', type=str, required=True,
                         help="Path to processed dataset (point this at the clean, "
                              "sigma=0.03, or sigma=0.05 noise-regime folder)")
    parser.add_argument('--model', type=str, default='rocbam', choices=['rocbam', 'robunet'],
                         help="Architecture to train: 'rocbam' or 'robunet' (baseline)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='models/checkpoints/best_model.pth')
    args = parser.parse_args()
    train(args)
