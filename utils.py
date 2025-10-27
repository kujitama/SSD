import torch
import torch.nn as nn
from torch.utils.data import Dataset
import timm
from torchvision import transforms
import torchvision.transforms.functional as F
import xml.etree.ElementTree as ET
import os
from PIL import Image
import numpy as np
import random

from ssd.augment import Compose, Resize, ToTensor, HorizontalFlip, RandomCrop, RandomExpand, Normalize, PhotometricDistort
from ssd.backbone import VGG16Backbone

# VOC class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class VocDataset(Dataset):
    def __init__(self, data_root, split='trainval', size=300, transform=None, years=['2007']):
        self.data_root = data_root
        self.split = split
        self.size = size
        self.transform = transform
        self.years = years if isinstance(years, list) else [years]
        
        # 複数年度のデータを統合
        self.image_ids = []
        self.year_mapping = []  # 各画像がどの年度のデータかを記録
        
        for year in self.years:
            split_file = os.path.join(data_root, f'VOC{year}', 'ImageSets', 'Main', f'{split}.txt')
            with open(split_file, 'r') as f:
                year_image_ids = [line.strip() for line in f.readlines()]
                self.image_ids.extend(year_image_ids)
                self.year_mapping.extend([year] * len(year_image_ids))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        year = self.year_mapping[idx]
        
        # 年度に応じたパスを設定
        image_dir = os.path.join(self.data_root, f'VOC{year}', 'JPEGImages')
        annotation_dir = os.path.join(self.data_root, f'VOC{year}', 'Annotations')
        
        # 画像の読み込み
        image_path = os.path.join(image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        # アノテーションの読み込み
        annotation_path = os.path.join(annotation_dir, f'{image_id}.xml')
        boxes, labels = self._parse_annotation(annotation_path)
        
        # バウンディングボックスの正規化
        orig_size = image.size  # (width, height)
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_size[0]  # x座標を正規化
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_size[1]  # y座標を正規化
            boxes = torch.FloatTensor(boxes)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.LongTensor(labels)
        
        # Transform適用（boxes対応）
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            # デフォルト変換（Resize + PIL→Tensor + 正規化）
            image = image.resize((self.size, self.size), Image.BILINEAR)
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id
        }
        
        return image, target
    
    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASS_TO_IDX:
                continue
            
            label = CLASS_TO_IDX[name]
            labels.append(label)
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, labels
    
class TrainTransform:
    def __init__(self, size=300, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = Compose([
            PhotometricDistort(),                      # 1. 色調補正
            RandomExpand(max_ratio=4.0, p=1),          # 2. Zoom out
            RandomCrop(                                # 3. Sample Crop
                target_size=size,
                min_scale=0.3,
                max_scale=1.0,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.0,
                max_attempts=50,
            ),
            HorizontalFlip(p=0.5),                     # 4. 水平反転
            Resize(size=size),                         # 5. リサイズ
            ToTensor(),                                # 6. Tensor変換
            Normalize(mean=mean, std=std)              # 7. 正規化
        ])
        
    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

class ValTransform:
    def __init__(self, size=300, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = Compose([
            Resize(size=size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

def build_backbone(cfg):
    backbone_name = cfg['name']
    pretrained = cfg.get('pretrained', True)
    
    if backbone_name == "vgg16":
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=False
        )
        return VGG16Backbone(backbone, backbone_name)
    else:
        raise NotImplementedError(f"Backbone {backbone_name} is not supported yet.")

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets
