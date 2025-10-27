import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes=None, labels=None):
        for transform in self.transforms:
            image, boxes, labels = transform(image, boxes, labels)
        
        return image, boxes, labels


class Resize:
    def __init__(self, size=300):
        self.size = size
    
    def __call__(self, image, boxes, labels):
        if isinstance(image, Image.Image):
            image = image.resize((self.size, self.size), Image.BILINEAR)
        else:
            raise ValueError("Resize transform only supports PIL Image.")
        
        # バウンディングボックスは正規化座標なので調整不要
        return image, boxes, labels


class ToTensor:
    def __call__(self, image, boxes, labels):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        return image, boxes, labels


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, boxes, labels):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, labels


class PhotometricDistort:
    def __call__(self, image, boxes, labels):
        if not isinstance(image, Image.Image):
            return image, boxes, labels
        
        # ランダムに処理順序を決定
        if random.random() < 0.5:
            # パターン1: Brightness → Contrast → Saturation → Hue
            image = self._adjust_brightness(image)
            image = self._adjust_contrast(image)
            image = self._adjust_saturation(image)
            image = self._adjust_hue(image)
        else:
            # パターン2: Brightness → Hue → Saturation → Contrast
            image = self._adjust_brightness(image)
            image = self._adjust_hue(image)
            image = self._adjust_saturation(image)
            image = self._adjust_contrast(image)
        
        return image, boxes, labels
    
    def _adjust_brightness(self, image):
        """明るさの調整"""
        if random.random() < 0.5:
            delta = random.uniform(-32, 32)
            enhancer = ImageEnhance.Brightness(image)
            factor = 1.0 + delta / 255.0
            factor = max(0, factor)
            image = enhancer.enhance(factor)
        return image
    
    def _adjust_contrast(self, image):
        """コントラストの調整"""
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        return image
    
    def _adjust_saturation(self, image):
        """彩度の調整"""
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        return image
    
    def _adjust_hue(self, image):
        """色相の調整"""
        if random.random() < 0.5:
            hue_shift = random.uniform(-18, 18)
            img = np.array(image.convert('HSV'), dtype=np.uint8)
            img[..., 0] = (img[..., 0].astype(int) + hue_shift) % 180
            image = Image.fromarray(img, mode='HSV').convert('RGB')

        return image
    
    
class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            # PIL画像を水平フリップ
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # バウンディングボックスも水平フリップ
            if len(boxes) > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]  # x座標を反転
        
        return image, boxes, labels


class RandomCrop:
    """
    論文中のSample Patchに準拠したIoU制約付きランダムクロップ
    - IoU閾値は {None, 0.1, 0.3, 0.5, 0.7, 0.9} から毎回抽選
    - パッチが採用される条件: (min_iou is None) or (少なくとも1つのGTとIoU>=min_iou)
    - ボックス採用条件: GTの中心がパッチ内
    - その後でパッチ座標に正規化再マップ → 最後にresize
    """
    def __init__(
        self,
        target_size=300,
        min_scale=0.1, max_scale=1.0,
        min_aspect_ratio=0.5, max_aspect_ratio=2.0,
        max_attempts=50,
        overlap_choices=(None, 0.1, 0.3, 0.5, 0.7, 0.9),
    ):
        self.target_size      = target_size
        self.min_scale        = min_scale
        self.max_scale        = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_attempts     = max_attempts
        self.overlap_choices  = overlap_choices

    def __call__(self, image, boxes, labels):
        if not isinstance(image, Image.Image):
            return image, boxes, labels
        if boxes is None or len(boxes) == 0:
            return image, boxes, labels

        width, height = image.size
        boxes_np = boxes.detach().cpu().numpy()

        # 今回のIoU下限を抽選
        min_iou = random.choice(self.overlap_choices)
        # Noneの場合は元画像を返す
        if min_iou is None:
            return image, boxes, labels

        for _ in range(self.max_attempts):
            # 面積スケールとアスペクト比をサンプル
            scale = random.uniform(self.min_scale, self.max_scale)
            ar    = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)

            crop_area = scale * width * height
            crop_h = int(np.sqrt(crop_area / ar))
            crop_w = int(np.sqrt(crop_area * ar))
            if crop_h <= 0 or crop_w <= 0 or crop_h > height or crop_w > width:
                continue

            # ランダムなクロップ位置を選択
            top  = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)

            # パッチ矩形の正規化座標
            cl = left / width
            ct = top / height
            cr = (left + crop_w) / width
            cb = (top + crop_h) / height

            # IoU条件（min_iouがある場合）を満たすか確認
            if min_iou is not None:
                # 各GTに対し IoU(rect_box, crop_rect) を計算
                # rect_box = boxes_np[i], crop_rect = [cl,ct,cr,cb]
                bx1 = boxes_np[:, 0]; by1 = boxes_np[:, 1]
                bx2 = boxes_np[:, 2]; by2 = boxes_np[:, 3]

                inter_x1 = np.maximum(bx1, cl)
                inter_y1 = np.maximum(by1, ct)
                inter_x2 = np.minimum(bx2, cr)
                inter_y2 = np.minimum(by2, cb)

                inter_w = np.clip(inter_x2 - inter_x1, 0.0, 1.0)
                inter_h = np.clip(inter_y2 - inter_y1, 0.0, 1.0)
                inter_area = inter_w * inter_h

                box_area  = np.clip((bx2 - bx1), 0.0, 1.0) * np.clip((by2 - by1), 0.0, 1.0)
                crop_area_n = (cr - cl) * (cb - ct)
                union_area = box_area + crop_area_n - inter_area
                ious = np.where(union_area > 0, inter_area / union_area, 0.0)

                # IoU条件を満たさない場合はやり直し
                if ious.max() < min_iou:
                    continue

            # 中心がパッチ内のボックスだけ採用
            cx = 0.5 * (boxes_np[:, 0] + boxes_np[:, 2])
            cy = 0.5 * (boxes_np[:, 1] + boxes_np[:, 3])
            keep = (cx >= cl) & (cx <= cr) & (cy >= ct) & (cy <= cb)
            if not np.any(keep):
                continue

            kept = boxes_np[keep]
            kept_labels = labels[keep] if isinstance(labels, torch.Tensor) else [labels[i] for i,k in enumerate(keep) if k]

            # パッチ座標系へ再マップ（正規化）
            denom_x = (cr - cl)
            denom_y = (cb - ct)
            # 交差領域でクリップしてからパッチ内に正規化
            x1 = np.clip(np.maximum(kept[:, 0], cl), cl, cr)
            y1 = np.clip(np.maximum(kept[:, 1], ct), ct, cb)
            x2 = np.clip(np.minimum(kept[:, 2], cr), cl, cr)
            y2 = np.clip(np.minimum(kept[:, 3], cb), ct, cb)

            new_x1 = (x1 - cl) / denom_x
            new_y1 = (y1 - ct) / denom_y
            new_x2 = (x2 - cl) / denom_x
            new_y2 = (y2 - ct) / denom_y

            valid_boxes = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)

            # 画像クロップ & リサイズ
            image_cropped = image.crop((left, top, left + crop_w, top + crop_h))
            image_resized = image_cropped.resize((self.target_size, self.target_size), Image.BILINEAR)

            new_boxes = torch.as_tensor(valid_boxes, dtype=boxes.dtype, device=boxes.device)
            if isinstance(labels, torch.Tensor):
                new_labels = kept_labels.to(labels.device)
            else:
                new_labels = torch.as_tensor(kept_labels, dtype=torch.long)

            return image_resized, new_boxes, new_labels

        # 採用できるパッチが見つからない場合は元を返す
        return image, boxes, labels


class RandomExpand:
    """
    論文中のZoom-Outに準拠したランダム拡張
    - 画像をランダム倍率(1〜max_ratio)のキャンバスに貼り付ける
    - ここではリサイズしない（後段のRandomCrop→最終リサイズで使う前提）
    """
    def __init__(
        self,
        max_ratio=16.0,
        fill=(123, 117, 104),
        p=0.5,
        boxes_normalized=True
    ):
        self.max_ratio = max_ratio
        self.fill = fill
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() > self.p:
            return image, boxes, labels
        if not isinstance(image, Image.Image):
            return image, boxes, labels
        if boxes is None or len(boxes) == 0:
            return image, boxes, labels

        w, h = image.size
        ratio = random.uniform(1.0, self.max_ratio)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        if new_w == w and new_h == h:
            return image, boxes, labels

        off_x = random.randint(0, new_w - w)
        off_y = random.randint(0, new_h - h)

        # 拡張キャンバスに貼り付け
        expanded = Image.new('RGB', (new_w, new_h), self.fill)
        expanded.paste(image, (off_x, off_y))

        if isinstance(boxes, torch.Tensor):
            b = boxes.clone()
        else:
            b = torch.as_tensor(boxes, dtype=torch.float32)

        # 拡張キャンバスに合わせてボックス座標を変換
        b[:, [0, 2]] = b[:, [0, 2]] * w + off_x
        b[:, [1, 3]] = b[:, [1, 3]] * h + off_y
        b[:, [0, 2]] = b[:, [0, 2]] / new_w
        b[:, [1, 3]] = b[:, [1, 3]] / new_h

        return expanded, b, labels