import torch
import torch.nn as nn
import torch.nn.functional as F

class BackFeature(nn.Module):
    def __init__(self, batch_size, num_classes, device):
        super(BackFeature, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device

    def forward_label(self, features, labels):
        features = F.normalize(features, dim=1)
        dist_matrix = torch.cdist(features, features, p=2)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.view(-1, 1)

        same_mask = torch.eq(labels, labels.T).float().to(self.device)
        same_mask.fill_diagonal_(0)
        diff_mask = 1. - same_mask
        diff_mask.fill_diagonal_(0)

        same_dist = dist_matrix * same_mask
        same_loss = same_dist.sum() / (same_mask.sum() + 1e-8)

        diff_dist = dist_matrix * diff_mask
        diff_loss = diff_dist.sum() / (diff_mask.sum() + 1e-8)

        total_loss = same_loss - diff_loss
        return total_loss
