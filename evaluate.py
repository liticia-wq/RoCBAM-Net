import argparse
import torch
from torch.utils.data import DataLoader
from models.roc_bam_net import RoCBAMNet
from models.losses import get_dice
from utils.dataset import BreastMultiPhaseDataset

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = BreastMultiPhaseDataset(args.data_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = RoCBAMNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    total_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += get_dice(outputs, masks).item()

    avg_test_dice = total_dice / len(test_loader)
    print(f" Final Mean Test Dice Score: {avg_test_dice:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    evaluate(args)
