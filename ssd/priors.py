import torch

def generate_priors(cfg):
    fm = cfg['priors']['feature_maps']
    steps = cfg['priors']['steps']
    min_sizes = cfg['priors']['min_sizes']
    max_sizes = cfg['priors']['max_sizes']
    aspect_ratios = cfg['priors']['aspect_ratios']
    image_size = cfg['input_size']

    anchors = []
    for k, f in enumerate(fm):
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) * steps[k] / image_size
                cy = (i + 0.5) * steps[k] / image_size

                s_k = min_sizes[k] / image_size
                anchors.append([cx, cy, s_k, s_k])
                s_k_prime = (min_sizes[k] * max_sizes[k]) ** 0.5 / image_size
                anchors.append([cx, cy, s_k_prime, s_k_prime])

                # 各アスペクト比のボックスを生成
                for ar in aspect_ratios[k]:
                    # sqrt(ar) : 1/sqrt(ar) = ar : 1
                    anchors.append([cx, cy, s_k * (ar**0.5), s_k/(ar**0.5)])
                    anchors.append([cx, cy, s_k/(ar**0.5), s_k * (ar**0.5)])
    priors = torch.tensor(anchors).clamp_(0, 1)  # [num_priors, 4], cx,cy,w,h (norm)
    return priors

def box_iou(boxes1, boxes2):
    # boxes: [N,4] in (xmin, ymin, xmax, ymax)
    tl = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # br < tlなら交差しないので0にclamp
    wh = (br - tl).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    area1 = (boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])
    iou = inter / (area1[:,None] + area2 - inter + 1e-7)
    return iou

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

def match(priors_cxcywh, targets_xyxy, labels, iou_threshold=0.5):
    pri_xyxy = cxcywh_to_xyxy(priors_cxcywh)
    iou = box_iou(targets_xyxy, pri_xyxy)     # [n_gt, n_priors]
    best_gt_iou, best_gt_idx = iou.max(dim=0) # 各priorに対するbest gt
    best_prior_iou, best_prior_idx = iou.max(dim=1) # 各gtに対するbest prior

    # 各GTに対して、必ず1つのpriorが割り当てられるようにする
    best_gt_iou[best_prior_idx] = 2.  # ensure > threshold
    best_gt_idx[best_prior_idx] = torch.arange(len(targets_xyxy), device=priors_cxcywh.device)

    conf = labels[best_gt_idx] + 1
    conf[best_gt_iou < iou_threshold] = 0  # background
    return best_gt_idx, conf