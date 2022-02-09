import torch
from torch import nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self, alpha=None, reduction='sum'):
        '''reduction: mean/sum'''
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        '''Calculate the focal loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
        '''
        device = inputs.device
        t = target
        p = inputs
        if t.size(0)==0:
            return torch.tensor(0).to(device).float()

        # w = alpha if t > 0 else 1-alpha
        if self.alpha is not None:
            w = self.alpha * t + (1 - self.alpha) * (1 - t)
        else:
            w = None

        return F.binary_cross_entropy(p, t, w, reduction=self.reduction)


class HierarchicalClass(nn.Module):
    def __init__(self, device):
        super(HierarchicalClass, self).__init__()
        self.device = device
        self.h1_loss_func = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0]).to(self.device))
        self.h2_loss_func = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0]).to(self.device))

    def forward(self, probas, labels):
        """
        probas: [B, C].
        labels: [B].
        """
        probas1, probas2 = probas
        index_01 = labels<2
        labels_2 = torch.ones_like(labels)
        labels_2[index_01] = 0
        h1_loss = self.h1_loss_func(probas1, labels_2)
        h2_loss = self.h2_loss_func(probas2, labels)

        loss = h1_loss + h2_loss

        return loss