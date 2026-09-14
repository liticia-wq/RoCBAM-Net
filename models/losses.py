import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, weight=50.0):
        super().__init__()
        self.weight = weight

    def forward(self, p, t):
        # 1. Calcul de la BCE standard
        # On utilise une petite valeur (epsilon) pour éviter les log(0)
        bce = F.binary_cross_entropy(p, t, reduction='none')

        # 2. On applique le poids manuellement sur les pixels positifs (la tumeur)
        # Cela force le modèle à ne pas ignorer les petites zones blanches
        weighted_bce = bce * (1 + t * (self.weight - 1))
        bce_loss = weighted_bce.mean()

        # 3. Calcul du Dice Loss (1 - Dice Score)
        inter = (p * t).sum()
        dice_coeff = (2. * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
        dice_loss = 1 - dice_coeff

        # Somme des deux pertes
        return bce_loss + dice_loss

def get_dice(p, t):
    # On seuille à 0.5 pour transformer les probabilités en masque binaire
    p = (p > 0.5).float()
    inter = (p * t).sum()
    return (2. * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)

