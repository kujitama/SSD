import warnings
warnings.filterwarnings('ignore')

import torch
import yaml
import random
import numpy as np
import csv
import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader
from ssd.priors import generate_priors, match
from ssd.loss import SSDLoss
from ssd.model import SSD300
from ssd.utils import encode
from utils import VocDataset, TrainTransform, ValTransform, collate_fn
from eval import evaluate_voc


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

def train(cfg_path, resume_from=None):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get('seed', 1337))
    
    exp_dir = os.path.basename(cfg_path).replace('.yaml', '')
    exp_dir = os.path.join("exp", exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_train = VocDataset(cfg['data_root'], split='trainval', size=cfg['input_size'], transform=TrainTransform(), years=cfg['dataset']['year'])
    ds_val   = VocDataset(cfg['data_root'], split='test',     size=cfg['input_size'], transform=ValTransform())
    dl_train = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=8)

    # デフォルトボックスの生成
    priors = generate_priors(cfg).to(device)  # [N,4]
    
    # aspect_ratiosから動的にnum_anchorsを計算
    aspect_ratios = cfg['priors']['aspect_ratios']
    num_anchors = [2 + 2*len(ar) for ar in aspect_ratios]  # min_size + max_size + 2*aspect_ratios
    
    model = SSD300(backbone=cfg['backbone'], num_anchors=num_anchors, num_classes=cfg['num_classes']).to(device)
    criterion = SSDLoss(cfg['hnm']['neg_pos_ratio'])

    if cfg['optimizer']['name'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=cfg['optimizer']['lr'],
                              momentum=cfg['optimizer']['momentum'],
                              weight_decay=cfg['optimizer']['weight_decay'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)

    best_map = 0.0
    start_epoch = 0
    
    # 既存モデルからの再開
    if resume_from:
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_map = checkpoint.get('best_map', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")
    
    # CSVログの初期化
    csv_filename = os.path.join(exp_dir, 'training_log.csv')
    csv_headers = ['epoch', 'loss', 'learning_rate', 'map50', 'timestamp']
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
    
    for epoch in tqdm(range(start_epoch, cfg['epochs']), desc='Training Epochs'):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, targets in dl_train:
            images = images.to(device)
            pred_locs, pred_confs = model(images)

            # make targets per-image
            target_locs_all, target_labels_all = [], []
            for t in targets:
                gt_xyxy, gt_labels = t['boxes'].to(device), t['labels'].to(device)
                
                # 
                matches, conf = match(priors, gt_xyxy, gt_labels, cfg['iou_threshold'])
                
                # encode offsets (cxcywh)
                # マッチしたGTボックスを使って回帰ターゲットを計算
                if len(gt_xyxy) > 0:
                    matched_gt_boxes = gt_xyxy[matches]  # [num_priors, 4]
                    target_locs = encode(matched_gt_boxes, priors)  # [num_priors, 4]
                else:
                    target_locs = torch.zeros((len(priors), 4), device=device)
                
                target_locs_all.append(target_locs)
                target_labels_all.append(conf)
            target_locs = torch.stack(target_locs_all)
            target_labels = torch.stack(target_labels_all)

            loss = criterion(pred_locs, pred_confs, target_locs, target_labels)
            
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{cfg["epochs"]}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # 毎エポック、損失と学習率をCSVに保存（mAP@0.5は評価時のみ）
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if (epoch+1) % cfg['eval_interval'] == 0:
            # mAPの簡易評価
            result = evaluate_voc(model, dl_val, priors, device)
            map50 = result['mAP']
            print(f'Epoch [{epoch+1}/{cfg["epochs"]}], mAP@0.5: {map50:.4f}')
            
            # mAP付きでCSVに保存
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch+1, f'{avg_loss:.6f}', f'{current_lr:.8f}', f'{map50:.6f}', timestamp])
            
            # ベストモデル保存
            if map50 > best_map:
                best_map = map50
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map': best_map,
                    'config': cfg
                }, os.path.join(exp_dir, f'best_model_{epoch+1}.pth'))
                print(f'New best model saved with mAP@0.5: {best_map:.4f}')
        else:
            # mAP評価なしの場合は空文字でCSVに保存
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch+1, f'{avg_loss:.6f}', f'{current_lr:.8f}', '', timestamp])

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    resume_from = None
    # resume_from = "exp/ssd300_voc_vgg16_005/best_model_300.pth"
    
    train(config_path, resume_from=resume_from)
