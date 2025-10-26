import torch
from typing import Tuple
from .priors import box_iou

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_th: float = 0.5, topk: int = 200) -> torch.Tensor:
    """
    Non-Maximum Suppression (NMS)
    
    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format
        scores: [N] tensor of confidence scores
        iou_th: IoU threshold for suppression
        topk: Maximum number of detections to keep
        
    Returns:
        Tensor of indices of kept detections
    """
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    if boxes.size(0) != scores.size(0):
        raise ValueError(f"Number of boxes ({boxes.size(0)}) must match number of scores ({scores.size(0)})")
    
    if boxes.size(1) != 4:
        raise ValueError(f"Boxes must have 4 coordinates, got {boxes.size(1)}")
    
    # スコアの高い順に見ていく
    idx = scores.argsort(descending=True)
    keep = []
    
    while idx.numel() > 0 and len(keep) < topk:
        i = idx[0]
        keep.append(i.item())

        # ボックスが1つだけ残ったら終了
        if idx.numel() == 1:
            break
        
        # NOTE: 1つずつ計算すると遅すぎるので何らかの工夫が必要
        # 最初にスコアの低すぎるものは弾くなど
        # torchvision.ops.nmsを使うのが楽
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idx[1:]])[0]
        idx = idx[1:][ious <= iou_th]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def decode(locs: torch.Tensor, priors: torch.Tensor, variances: Tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """
    Decode SSD predictions to bounding boxes
    
    Args:
        locs: [N, 4] predicted offsets [dx, dy, dw, dh]
        priors: [N, 4] prior boxes in (cx, cy, w, h) normalized format
        variances: Variance values for coordinate and size decoding
        
    Returns:
        Decoded boxes in (x1, y1, x2, y2) format
    """
    if locs.size() != priors.size():
        raise ValueError(f"locs size {locs.size()} must match priors size {priors.size()}")
    
    if locs.size(1) != 4:
        raise ValueError(f"Both locs and priors must have 4 coordinates, got {locs.size(1)}")
    
    # Decode center coordinates and sizes
    # priors format: (cx, cy, w, h) normalized
    # locs format: (dx, dy, dw, dh) offsets
    decoded_centers = priors[:, :2] + locs[:, :2] * variances[0] * priors[:, 2:]
    decoded_sizes = priors[:, 2:] * torch.exp(locs[:, 2:] * variances[1])
    
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    cx, cy = decoded_centers.unbind(-1)
    w, h = decoded_sizes.unbind(-1)
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    # Clamp coordinates to [0, 1] range (normalized coordinates)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes.clamp_(0, 1)
