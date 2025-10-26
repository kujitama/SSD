import torch
import torch.nn as nn

class MultiBoxHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loc = nn.ModuleList()
        self.cls = nn.ModuleList()
        for c, k in zip(in_channels, num_anchors):
            self.loc.append(nn.Conv2d(c, k*4, kernel_size=3, padding=1))
            self.cls.append(nn.Conv2d(c, k*num_classes, kernel_size=3, padding=1))
            
    def forward(self, features):
        locs, confs = [], []
        for x, l , c in zip(features, self.loc, self.cls):
            # contiguous()のchatgpt解説: https://chatgpt.com/share/68e21c41-b620-8004-a85a-cc56670e4ed3
            locs.append(l(x).permute(0,2,3,1).contiguous())
            confs.append(c(x).permute(0,2,3,1).contiguous())
        locs = torch.cat([o.view(o.size(0), -1, 4) for o in locs], dim=1)
        confs = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in confs], dim=1)
        return locs, confs
