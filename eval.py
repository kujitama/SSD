import torch
from torch.nn.functional import softmax
import numpy as np
from tqdm import tqdm
from ssd.utils import decode
import torchvision.ops
from utils import VOC_CLASSES, ValTransform
import wandb
import json

def evaluate_voc(model, dataloader, priors, device, conf_threshold=0.05, nms_threshold=0.45, max_detections=200, debug=False):
    """
    PASCAL VOCデータセットでのmAP評価
    
    Args:
        model: SSDモデル
        dataloader: 検証用データローダー
        priors: Prior boxes
        device: デバイス
        conf_threshold: 信頼度閾値
        nms_threshold: NMS閾値
    
    Returns:
        結果辞書：{
            'mAP': float,
            'class_aps': list,
            'class_ap_dict': dict
        }
    """
    model.eval()
    
    all_detections = []
    all_annotations = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            batch_size = images.size(0)
            
            # 推論
            pred_locs, pred_confs = model(images)
            
            for i in range(batch_size):
                # バッチ内の各画像について処理
                loc = pred_locs[i]  # [num_priors, 4]
                conf = pred_confs[i]  # [num_priors, num_classes]
                
                # デコード
                decoded_boxes = decode(loc, priors)
                
                # 座標を[0, 1]の範囲にクリップ
                decoded_boxes = torch.clamp(decoded_boxes, 0.0, 1.0)
                
                # ロジットを確率に変換（ソフトマックス適用）
                conf_probs = softmax(conf, dim=-1)
                
                # 信頼度でフィルタリング（全クラス一括、背景クラス除く）
                max_conf_scores, max_conf_classes = conf_probs[:, 1:].max(dim=1)  # 背景クラスを除く
                conf_mask = max_conf_scores > conf_threshold
                
                image_detections = []
                if conf_mask.sum() > 0:
                    # フィルタリング後のボックスとスコア
                    filtered_boxes = decoded_boxes[conf_mask]
                    filtered_scores = max_conf_scores[conf_mask]
                    # max_conf_classesは0-19の範囲（背景除く）なので、+1して1-20にする
                    filtered_classes = max_conf_classes[conf_mask] + 1  # VOCクラス1-20に変換
                    
                    # TOP-Kフィルタリング（処理量削減）
                    if len(filtered_scores) > max_detections:
                        top_k_indices = torch.topk(filtered_scores, max_detections)[1]
                        filtered_boxes = filtered_boxes[top_k_indices]
                        filtered_scores = filtered_scores[top_k_indices]
                        filtered_classes = filtered_classes[top_k_indices]
                    
                    # batched_nmsを使用（クラス別NMSを一度に処理）
                    keep = torchvision.ops.batched_nms(
                        filtered_boxes, 
                        filtered_scores, 
                        filtered_classes, 
                        nms_threshold
                    )
                    
                    if len(keep) > 0:
                        final_boxes = filtered_boxes[keep]
                        final_scores = filtered_scores[keep]
                        final_classes = filtered_classes[keep]
                        
                        for j in range(len(keep)):
                            image_detections.append({
                                'bbox': final_boxes[j].cpu().numpy(),
                                'score': final_scores[j].item(),
                                'class_id': final_classes[j].item()
                            })
                
                all_detections.append(image_detections)
                
                # Ground truthの準備
                target = targets[i]
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                image_annotations = []
                if len(gt_boxes) > 0:
                    for j in range(len(gt_boxes)):
                        image_annotations.append({
                            'bbox': gt_boxes[j],
                            'class_id': gt_labels[j] + 1,  # 0-19 → 1-20に変換（VOCクラス）
                            'difficult': False  # 簡略化のため
                        })
                
                all_annotations.append(image_annotations)
    
    # mAP計算
    map50, class_aps = calculate_map(all_detections, all_annotations, iou_threshold=0.5, debug=debug)
    
    # クラス別APを辞書形式に変換
    class_ap_dict = {}
    for class_id, ap in enumerate(class_aps):
        class_name = VOC_CLASSES[class_id]
        class_ap_dict[class_name] = float(ap)
    
    return {
        'mAP': map50,
        'class_aps': class_aps,
        'class_ap_dict': class_ap_dict
    }

def calculate_map(detections, annotations, iou_threshold=0.5, debug=False):
    """
    mAP計算
    
    Args:
        detections: 検出結果のリスト
        annotations: Ground truthのリスト  
        iou_threshold: IoU閾値
    
    Returns:
        (map_score, aps): mAP値と各クラスのAPのリスト
    """
    num_classes = len(VOC_CLASSES)
    aps = []
    
    if debug:
        print(f"\n=== mAP計算開始 ===")
        print(f"総画像数: {len(detections)}")
        print(f"クラス数: {num_classes}")
    
    for class_id in range(1, num_classes + 1):  # 背景クラスを除く
        # クラス別の検出結果とGTを収集
        class_detections = []
        class_annotations = []
        
        for img_idx, (img_dets, img_anns) in enumerate(zip(detections, annotations)):
            # 検出結果
            for det in img_dets:
                if det['class_id'] == class_id:
                    class_detections.append({
                        'image_id': img_idx,
                        'bbox': det['bbox'],
                        'score': det['score']
                    })
            
            # Ground truth
            for ann in img_anns:
                if ann['class_id'] == class_id:
                    class_annotations.append({
                        'image_id': img_idx,
                        'bbox': ann['bbox'],
                        'difficult': ann.get('difficult', False)
                    })
        
        if len(class_detections) == 0:
            aps.append(0.0)
            if debug:
                class_name = VOC_CLASSES[class_id - 1]
                print(f"クラス {class_id} ({class_name}): 検出なし, AP = 0.0")
            continue
        
        # スコアで降順ソート
        class_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # AP計算
        ap = calculate_ap(class_detections, class_annotations, iou_threshold)
        aps.append(ap)
        
        if debug:
            class_name = VOC_CLASSES[class_id - 1]
            print(f"クラス {class_id} ({class_name}): 検出={len(class_detections)}, GT={len(class_annotations)}, AP={ap:.4f}")
    
    mean_ap = np.mean(aps)
    
    if debug:
        print(f"\n=== mAP計算結果 ===")
        print(f"各クラスのAP: {[f'{ap:.4f}' for ap in aps]}")
        print(f"平均mAP: {mean_ap:.4f}")
    
    return mean_ap, aps

def calculate_ap(detections, annotations, iou_threshold):
    """
    単一クラスのAP計算
    """
    if len(detections) == 0 or len(annotations) == 0:
        return 0.0
    
    # GTをimage_id別に整理し、マッチ状態も管理
    gt_by_image = {}
    gt_matched = {}  # GTがマッチ済みかどうかを管理
    
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
            gt_matched[img_id] = []
        gt_by_image[img_id].append(ann)
        gt_matched[img_id].append(False)  # 初期状態は未マッチ
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    for det_idx, det in enumerate(detections):
        img_id = det['image_id']
        det_bbox = det['bbox']
        
        if img_id in gt_by_image:
            gt_bboxes = [ann['bbox'] for ann in gt_by_image[img_id]]
            
            if len(gt_bboxes) > 0:
                # IoU計算（torchvision.ops.box_iouを使用）
                det_tensor = torch.tensor([det_bbox], dtype=torch.float32)
                gt_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)
                ious = torchvision.ops.box_iou(det_tensor, gt_tensor)[0]  # [num_gt]
                
                max_iou, max_idx = ious.max(dim=0)
                max_idx = max_idx.item()
                
                # IoU閾値を満たし、かつそのGTがまだマッチしていない場合のみTP
                if max_iou >= iou_threshold and not gt_matched[img_id][max_idx]:
                    tp[det_idx] = 1
                    gt_matched[img_id][max_idx] = True  # GTをマッチ済みにする
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1
        else:
            fp[det_idx] = 1
    
    # Precision-Recallカーブ計算
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    num_annotations = len(annotations)
    recalls = tp_cumsum / max(num_annotations, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # AP計算（全点補間）
    # precisionを右から左へ補間（各点で、それ以降の最大値を保持）
    precisions_interp = np.zeros_like(precisions)
    max_prec = 0.0
    for i in range(len(precisions) - 1, -1, -1):
        max_prec = max(max_prec, precisions[i])
        precisions_interp[i] = max_prec
    
    # recall変化点での面積を計算
    recalls_with_start = np.concatenate([[0.0], recalls])
    precisions_with_start = np.concatenate([[precisions_interp[0]], precisions_interp])
    
    ap = 0.0
    for i in range(len(recalls)):
        ap += (recalls_with_start[i + 1] - recalls_with_start[i]) * precisions_with_start[i + 1]
    
    return ap

if __name__ == '__main__':
    import argparse
    import yaml
    import os
    from torch.utils.data import DataLoader
    from ssd.priors import generate_priors
    from ssd.model import SSD300
    from utils import VocDataset,  collate_fn
    
    parser = argparse.ArgumentParser(description='SSD評価スクリプト')
    parser.add_argument('--config', default='configs/ssd300_voc_vgg16.yaml', help='設定ファイルパス')
    parser.add_argument('--model', default='best_model.pth', help='モデルファイルパス')
    parser.add_argument('--visualize', action='store_true', help='結果を可視化')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='可視化サンプル数')
    parser.add_argument('--vis_conf_threshold', type=float, default=0.3, help='可視化用信頼度閾値')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報表示')
    
    args = parser.parse_args()
    
    # configファイル名から実験名を抽出
    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    
    # Wandbで記録
    wandb.init(
        project='SSD_VOCtest',
        name=config_filename,
        config={
            'config_file': args.config,
            'model_file': args.model,
            'conf_threshold': 0.05,
            'nms_threshold': 0.45
        }
    )
    
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
    
    dl_val = DataLoader(
        ds_val, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=8
    )
    
    print(f"検証データ数: {len(ds_val)}")
    
    # デフォルトボックス生成
    priors = generate_priors(cfg).to(device)
    print(f"デフォルトボックス数: {len(priors)}")

    # モデル構築
    aspect_ratios = cfg['priors']['aspect_ratios']
    num_anchors = [2 + 2*len(ar) for ar in aspect_ratios]
    
    model = SSD300(
        cfg['backbone'], 
        num_anchors=num_anchors, 
        num_classes=cfg['num_classes']
    ).to(device)
    
    # 学習済みモデル読み込み
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"学習済みモデル読み込み完了 (エポック: {checkpoint['epoch']}, ベストmAP: {checkpoint.get('best_map', 'N/A')})")
    except FileNotFoundError:
        print(f"エラー: {args.model}が見つかりません。")
        exit(1)
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        exit(1)
    
    # 評価実行
    print("\n=== mAP評価開始 ===")
    results = evaluate_voc(
        model, 
        dl_val, 
        priors, 
        device, 
        conf_threshold=0.05,
        nms_threshold=0.45,
        debug=args.debug  # デバッグ情報表示
    )
    
    map50 = results['mAP']
    class_ap_dict = results['class_ap_dict']
    
    print(f"\n最終結果 mAP@0.5: {map50:.4f}")
    
    # 結果をJSON形式で保存
    results_json = {
        'experiment': config_filename,
        'mAP': float(map50),
        'class_APs': class_ap_dict
    }

    with open(f'exp/{config_filename}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # Wandbにログを送信
    wandb.log({
        'mAP': map50,
        **{f'AP_{class_name}': ap for class_name, ap in class_ap_dict.items()}
    })
    
    # 可視化実行
    if args.visualize:
        print("\n=== 結果可視化開始 ===")
        from visualize_results import visualize_predictions
        
        # 可視化用に小さいバッチサイズでデータローダー作成
        dl_vis = DataLoader(
            ds_val, 
            batch_size=4, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        visualize_predictions(
            model, 
            dl_vis, 
            priors, 
            device, 
            num_samples=args.num_vis_samples,
            conf_threshold=args.vis_conf_threshold
        )
    
    wandb.finish()
