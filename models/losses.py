import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, weight=50.0):
        super(WeightedBCEDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        pred = torch.sigmoid(inputs)
        smooth = 1e-5
        intersection = (pred * targets).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        return bce + (self.weight * dice_loss)

def get_dice(inputs, targets):
    pred = torch.sigmoid(inputs)
    pred = (pred > 0.5).float()
    smooth = 1e-5
    intersection = (pred * targets).sum()
    return (2. * intersection + smooth) / (pred.sum() + targets.sum() + smooth)
