import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def get_batch_best_mode(gt, predictions):
    '''
    input:
        - gt: [B, L, 2]
        - predictions: [B, K, L, 2]
    return:
        - [B]
    '''
    gt = torch.unsqueeze(gt, 1) #
    l2_norm = (torch.norm(predictions - gt, p=2, dim=-1)).sum(dim=-1)
    best_reg = l2_norm.argmin(axis=-1)

    return best_reg

class TrajLoss(nn.Module):
    def __init__(self):
        super(TrajLoss, self).__init__()
        
    def forward(self, gt, predictions, confidences):

        best_reg = get_batch_best_mode(gt, predictions)
        cls_loss = F.cross_entropy(nn.functional.log_softmax(confidences, dim=1), best_reg)
        
        best_predictions = predictions[torch.arange(gt.shape[0]), best_reg]
        reg_loss = torch.norm(gt.squeeze(1) - best_predictions, p=2, dim=-1).mean()
        loss = reg_loss + cls_loss
        
        return loss
