import torch
import torch.nn as nn

from ssd.backbone import build_backbone
from .heads import MultiBoxHead

def l2norm():
    class L2Norm(nn.Module):
        def __init__(self, n_channels, scale=20):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(n_channels)*scale)
            
        def forward(self, x):
            return (x / x.pow(2).sum(dim=1, keepdim=True).sqrt()) * self.weight.view(1, -1, 1, 1)
        
    return L2Norm(512, scale=20)

class SSD300(nn.Module):
    def __init__(self, backbone, num_anchors, num_classes):
        super().__init__()
        self.backbone = build_backbone(backbone)
        
        if backbone['name'] == 'vgg16':
            self.l2norm = l2norm()
            in_channels = [512, 1024, 512, 256, 256, 256]
            self.head = MultiBoxHead(in_channels, num_anchors, num_classes)
        else:
            raise NotImplementedError(f"Backbone {backbone['name']} is not supported yet.")

    def forward(self, x):
        feats = self.backbone(x)
        feats[0] = self.l2norm(feats[0])
        locs, confs = self.head(feats)
        return locs, confs  # [B, num_priors, 4], [B, num_priors, num_classes]