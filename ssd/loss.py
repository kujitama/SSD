import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, pred_locs, pred_confs, target_locs, target_labels):
        # pred_confs: [B,N,C], pred_locs: [B,N,4]
        num_classes = pred_confs.size(-1)
        pos_mask = target_labels > 0  # positives
        # 回帰: posのみ Smooth L1
        loc_loss = F.smooth_l1_loss(pred_locs[pos_mask], target_locs[pos_mask], reduction='sum')

        # 分類: まず全priorでクロスエントロピー
        conf_loss_all = F.cross_entropy(pred_confs.view(-1, num_classes), target_labels.view(-1), reduction='none').view(pred_confs.size(0), -1)

        # ----- Hard Negative Mining -----
        # ポジティブを除いた中からネガティブを選ぶ
        conf_loss_pos = conf_loss_all[pos_mask]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_mask] = 0  # ignore positives

        # ネガティブ数の制限
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # [B,1]
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos_mask.size(1)-1)

        conf_loss_neg_sorted, idx = conf_loss_neg.sort(dim=1, descending=True)
        neg_mask = torch.zeros_like(conf_loss_neg, dtype=torch.bool)
        for i in range(pred_confs.size(0)):
            n = int(num_neg[i])
            if n > 0:
                neg_mask[i, idx[i, :n]] = True

        conf_loss = (conf_loss_all[pos_mask].sum() + conf_loss_all[neg_mask].sum())

        N = num_pos.sum().clamp(min=1).float()
        return (loc_loss + conf_loss) / N
