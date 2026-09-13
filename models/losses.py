import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEDiceLoss(nn.Module):
    """
    Fonction de perte hybride pour le modèle RoCBAM-Net sur le dataset BreastDM.
    Combine une Entropie Croisée Binaire pondérée (L_wBCE) avec un poids w=50
    et une Perte de Dice (L_Dice) avec un terme de lissage epsilon=1e-6.
    """
    def __init__(self, weight=50.0, smooth=1e-6):
        super(WeightedBCEDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Conversion des logits en probabilités p_i via Sigmoid
        probs = torch.sigmoid(inputs)
        
        # 1. Calcul de la perte Weighted BCE (L_wBCE)
        # Formule : L_wBCE = - (1/N) * sum( w * t_i * log(p_i) + (1 - t_i) * log(1 - p_i) )
        bce_loss = -(
            self.weight * targets * torch.log(probs + 1e-7) + 
            (1.0 - targets) * torch.log(1.0 - probs + 1e-7)
        )
        l_wbce = torch.mean(bce_loss)

        # 2. Calcul de la perte de Dice (L_Dice)
        # Formule : L_Dice = 1 - (2 * sum(p_i * t_i) + eps) / (sum(p_i) + sum(t_i) + eps)
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs) + torch.sum(targets)
        l_dice = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth))

        # 3. Perte Totale : L_total = L_wBCE + L_Dice
        return l_wbce + l_dice

def get_dice(inputs, targets, smooth=1e-6):
    """
    Calcule le coefficient de similitude Dice (DSC) binaire pour l'évaluation.
    """
    probs = torch.sigmoid(inputs)
    preds = (probs > 0.5).float()
    
    intersection = torch.sum(preds * targets)
    cardinality = torch.sum(preds) + torch.sum(targets)
    
    return ((2.0 * intersection + smooth) / (cardinality + smooth))
