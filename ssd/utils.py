import torch
import torch.nn.functional as F

def encode(gt_boxes, priors, variances=[0.1, 0.2]):
    """
    バウンディングボックスをSSD形式のオフセットにエンコード
    
    Args:
        gt_boxes: Ground truth boxes in xyxy format [N, 4]
        priors: Prior boxes in cxcywh format [M, 4]
        variances: Variance values for encoding [2]
    
    Returns:
        encoded: Encoded offsets [M, 4] in (dx, dy, dw, dh) format
    """
    # gt_boxesをcxcywh形式に変換
    gt_cxcy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # center x, y
    gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]  # width, height
    
    # Prior boxesの中心とサイズ
    prior_cxcy = priors[:, :2]
    prior_wh = priors[:, 2:]
    
    # オフセットの計算
    # Center offset: (gt_center - prior_center) / prior_wh / variance[0]
    encoded_cxcy = (gt_cxcy - prior_cxcy) / prior_wh / variances[0]
    
    # Size offset: log(gt_wh / prior_wh) / variance[1] 
    encoded_wh = torch.log(gt_wh / prior_wh) / variances[1]
    
    return torch.cat([encoded_cxcy, encoded_wh], dim=1)

def decode(pred_locs, priors, variances=[0.1, 0.2]):
    """
    SSD形式のオフセットをバウンディングボックスにデコード
    
    Args:
        pred_locs: Predicted offsets [N, 4] in (dx, dy, dw, dh) format
        priors: Prior boxes in cxcywh format [N, 4]
        variances: Variance values for decoding [2]
    
    Returns:
        decoded: Decoded boxes in xyxy format [N, 4]
    """
    # Prior boxesの中心とサイズ
    prior_cxcy = priors[:, :2]
    prior_wh = priors[:, 2:]
    
    # 予測されたオフセットからバウンディングボックスを復元
    # Center: prior_center + pred_offset * prior_wh * variance[0]
    decoded_cxcy = pred_locs[:, :2] * variances[0] * prior_wh + prior_cxcy
    
    # Size: prior_wh * exp(pred_offset * variance[1])
    decoded_wh = torch.exp(pred_locs[:, 2:] * variances[1]) * prior_wh
    
    # xyxy形式に変換
    decoded_boxes = torch.cat([
        decoded_cxcy - decoded_wh / 2,  # xmin, ymin
        decoded_cxcy + decoded_wh / 2   # xmax, ymax
    ], dim=1)
    
    return decoded_boxes

def xyxy_to_cxcywh(boxes):
    """
    xyxy形式のバウンディングボックスをcxcywh形式に変換
    
    Args:
        boxes: Boxes in xyxy format [N, 4]
    
    Returns:
        boxes_cxcywh: Boxes in cxcywh format [N, 4]
    """
    cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([cxcy, wh], dim=1)

def cxcywh_to_xyxy(boxes):
    """
    cxcywh形式のバウンディングボックスをxyxy形式に変換
    
    Args:
        boxes: Boxes in cxcywh format [N, 4]
    
    Returns:
        boxes_xyxy: Boxes in xyxy format [N, 4]
    """
    xy_min = boxes[:, :2] - boxes[:, 2:] / 2
    xy_max = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat([xy_min, xy_max], dim=1)
