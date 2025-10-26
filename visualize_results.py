#!/usr/bin/env python3
"""
SSD検出結果の可視化スクリプト
"""
import torch
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from ssd.priors import generate_priors
from ssd.model import SSD300
from ssd.utils import decode
from utils import ValTransform, VocDataset, build_backbone, collate_fn, VOC_CLASSES
import torchvision.ops
import torch.nn.functional as F
from tqdm import tqdm

# 色の定義（クラス別）
COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#FFA500', '#A52A2A', '#DDA0DD', '#98FB98', '#F0E68C', '#DEB887',
    '#D2691E', '#FF1493'
]

def create_results_folder():
    """結果保存用フォルダを作成"""
    results_dir = 'results'
    subdirs = ['detections', 'comparisons', 'grid_samples']
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for subdir in subdirs:
        path = os.path.join(results_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    return results_dir

def denormalize_image(image_tensor):
    """正規化された画像テンソルを元の値域に戻す"""
    # [0, 1] の範囲のテンソルを [0, 255] の範囲に変換
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image

def draw_bboxes_on_image(image_array, detections, gt_annotations=None, save_path=None, image_id=None):
    """
    画像上にバウンディングボックスを描画
    
    Args:
        image_array: numpy array [H, W, 3]
        detections: 検出結果のリスト
        gt_annotations: Ground truth (オプション)
        save_path: 保存パス
        image_id: 画像ID
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_array)
    
    # 検出結果を描画（赤色）
    for det in detections:
        bbox = det['bbox']
        score = det['score']
        class_id = det['class_id']
        class_name = VOC_CLASSES[class_id - 1]  # class_idは1-20なので-1
        
        # 正規化座標を画像座標に変換
        h, w = image_array.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        
        # バウンディングボックス描画
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # ラベル描画
        label = f'{class_name}: {score:.2f}'
        ax.text(x1, y1 - 5, label, color='red', fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Ground truth描画（緑色、提供された場合）
    if gt_annotations:
        for gt in gt_annotations:
            bbox = gt['bbox']
            class_id = gt['class_id']
            class_name = VOC_CLASSES[class_id - 1]
            
            # 正規化座標を画像座標に変換
            h, w = image_array.shape[:2]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            
            # バウンディングボックス描画
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            # ラベル描画
            label = f'GT: {class_name}'
            ax.text(x1, y2 + 15, label, color='green', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    ax.set_title(f'Detection Results - Image {image_id}' if image_id else 'Detection Results')
    ax.axis('off')
    
    # 凡例追加
    if gt_annotations:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Predictions'),
            Line2D([0], [0], color='green', lw=2, linestyle='--', label='Ground Truth')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可視化画像を保存: {save_path}")
    
    plt.close()

def create_grid_visualization(images, all_detections, all_annotations, save_path, num_images=9):
    """グリッド形式で複数画像の結果を表示"""
    rows = int(np.sqrt(num_images))
    cols = (num_images + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    if rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx in range(num_images):
        if idx >= len(images):
            break
            
        row = idx // cols
        col = idx % cols
        
        ax = axes[row][col] if rows > 1 else axes[col]
        
        # 画像表示
        image_array = denormalize_image(images[idx])
        ax.imshow(image_array)
        
        # 検出結果描画
        detections = all_detections[idx]
        annotations = all_annotations[idx]
        
        h, w = image_array.shape[:2]
        
        # 予測結果（赤）
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        # Ground truth（緑）
        for gt in annotations:
            bbox = gt['bbox']
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Image {idx+1}')
        ax.axis('off')
    
    # 空のサブプロットを非表示
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"グリッド可視化を保存: {save_path}")
    plt.close()

def visualize_predictions(model, dataloader, priors, device, num_samples=10, conf_threshold=0.3):
    """
    予測結果を可視化
    
    Args:
        model: 学習済みモデル
        dataloader: データローダー
        priors: デフォルトボックス
        device: デバイス
        num_samples: 可視化するサンプル数
        conf_threshold: 信頼度閾値
    """
    model.eval()
    results_dir = create_results_folder()
    
    all_images = []
    all_detections = []
    all_annotations = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc='可視化用データ処理')):
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            batch_size = images.size(0)
            
            # 推論
            pred_locs, pred_confs = model(images)
            
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                # 画像とターゲット取得
                image = images[i]
                target = targets[i]
                
                # 予測処理
                loc = pred_locs[i]
                conf = pred_confs[i]
                
                # デコード
                decoded_boxes = decode(loc, priors)
                decoded_boxes = torch.clamp(decoded_boxes, 0.0, 1.0)
                
                # ロジットを確率に変換（ソフトマックス適用）
                conf_probs = F.softmax(conf, dim=-1)
                
                # 信頼度フィルタリング（背景クラス除く）
                max_conf_scores, max_conf_classes = conf_probs[:, 1:].max(dim=1)
                conf_mask = max_conf_scores > conf_threshold
                
                detections = []
                if conf_mask.sum() > 0:
                    filtered_boxes = decoded_boxes[conf_mask]
                    filtered_scores = max_conf_scores[conf_mask]
                    filtered_classes = max_conf_classes[conf_mask] + 1
                    
                    # NMS
                    keep = torchvision.ops.batched_nms(
                        filtered_boxes, filtered_scores, filtered_classes, 0.45
                    )
                    
                    if len(keep) > 0:
                        final_boxes = filtered_boxes[keep]
                        final_scores = filtered_scores[keep]
                        final_classes = filtered_classes[keep]
                        
                        for j in range(len(keep)):
                            detections.append({
                                'bbox': final_boxes[j].cpu().numpy(),
                                'score': final_scores[j].item(),
                                'class_id': final_classes[j].item()
                            })
                
                # Ground truth準備
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                annotations = []
                
                if len(gt_boxes) > 0:
                    for j in range(len(gt_boxes)):
                        annotations.append({
                            'bbox': gt_boxes[j],
                            'class_id': gt_labels[j] + 1
                        })
                
                # データ保存
                all_images.append(image)
                all_detections.append(detections)
                all_annotations.append(annotations)
                
                # 個別画像保存
                image_array = denormalize_image(image)
                save_path = os.path.join(results_dir, 'comparisons', f'image_{sample_count:03d}.png')
                draw_bboxes_on_image(
                    image_array, detections, annotations, 
                    save_path, f"{sample_count:03d}"
                )
                
                sample_count += 1
    
    # グリッド可視化作成
    if len(all_images) >= 4:
        grid_save_path = os.path.join(results_dir, 'grid_samples', 'detection_grid.png')
        create_grid_visualization(
            all_images[:9], all_detections[:9], all_annotations[:9], 
            grid_save_path, min(9, len(all_images))
        )
    
    # 統計情報保存
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"可視化統計情報\n")
        f.write(f"{'='*50}\n")
        f.write(f"処理画像数: {len(all_images)}\n")
        f.write(f"信頼度閾値: {conf_threshold}\n")
        f.write(f"平均検出数: {np.mean([len(det) for det in all_detections]):.2f}\n")
        f.write(f"平均GT数: {np.mean([len(ann) for ann in all_annotations]):.2f}\n")
        
        # クラス別統計
        f.write(f"\nクラス別検出数:\n")
        class_counts = {}
        for detections in all_detections:
            for det in detections:
                class_id = det['class_id']
                class_name = VOC_CLASSES[class_id - 1]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            f.write(f"  {class_name}: {count}\n")
    
    print(f"\n可視化完了!")
    print(f"結果保存先: {results_dir}/")
    print(f"- 個別画像: {results_dir}/comparisons/")
    print(f"- グリッド表示: {results_dir}/grid_samples/")
    print(f"- 統計情報: {summary_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SSD検出結果可視化')
    parser.add_argument('--config', default='configs/ssd300_voc_vgg16.yaml', help='設定ファイル')
    parser.add_argument('--model', default='best_model.pth', help='モデルファイル')
    parser.add_argument('--num_samples', type=int, default=10, help='可視化サンプル数')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='信頼度閾値')
    
    args = parser.parse_args()
    
    # 設定読み込み
    cfg = yaml.safe_load(open(args.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"デバイス: {device}")
    
    # データセット準備
    ds_val = VocDataset(
        cfg['data_root'],
        split='test',
        size=cfg['input_size'], 
        transform=ValTransform()
    )
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # モデル準備
    priors = generate_priors(cfg).to(device)
    aspect_ratios = cfg['priors']['aspect_ratios']
    num_anchors = [2 + 2*len(ar) for ar in aspect_ratios]
    
    backbone = build_backbone(cfg['backbone'])
    model = SSD300(backbone, None, num_anchors, cfg['num_classes']).to(device)
    
    # モデル読み込み
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデル読み込み完了: {args.model}")
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        exit(1)
    
    # 可視化実行
    visualize_predictions(
        model, dl_val, priors, device, 
        args.num_samples, args.conf_threshold
    )
