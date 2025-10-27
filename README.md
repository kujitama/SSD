# SSD (Single Shot MultiBox Detector) 論文実装

SSD300 (Weislin Liu et al., 2016) の PyTorch による実装です。
論文の再現実装を通して、SSDの理解を深めることを目的としています。

## 概要

本実装は以下の論文に基づいています：
- **論文**: SSD: Single Shot MultiBox Detector
- **著者**: Wei Liu, Dragomir Anguelov, Dumitru Erhan, et al.
- **リンク**: [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)


## 実装内容

### コアコンポーネント
- **バックボーン**: VGG16, ResNet50, MobileNetV2, EfficientNetV2 に対応
- **マルチスケール特徴マップ**: 6層の異なるスケールから予測
- **デフォルトボックス（Prior Boxes）**: 複数のアスペクト比と異なるスケール
- **損失関数**: クラス分類損失 (Softmax) + 位置回帰損失 (Smooth L1) + Hard Negative Mining
- **後処理**: Non-Maximum Suppression (NMS)
- **Data Augmentation**: Sample Patch、Zoom Outなど（論文準拠）

### ディレクトリ構成
```
ssd/
├── model.py          # SSD300 モデル定義
├── backbone.py       # バックボーン実装
├── heads.py          # SSD ヘッド（予測層）
├── loss.py           # 損失関数と Hard Negative Mining
├── priors.py         # デフォルトボックス生成
├── augment.py        # データ拡張
└── utils.py          # ユーティリティ関数（encode/decode）

configs/               # 設定ファイル（YAML）
exp/                  # 実験結果ディレクトリ
```


## 使用方法

### 学習

設定ファイルを指定して学習を実行：
```bash
python train.py configs/ssd300_voc_vgg16.yaml
```

### 評価

学習済みモデルで評価を実行：
```bash
python eval.py --config configs/ssd300_voc_vgg16.yaml \
                --model exp/ssd300_voc_vgg16/best_model_X.pth
```

## 実験結果

### 実験サマリー

| No | mAP@0.5  | 備考 |
|----|----------|------|
| 1  | **0.490** | 初期実装 |
| 2  | **0.587** | Augmentationを強化（RandomCrop、RandomExpandを追加）|
| 3  | **0.659** | VOC2012データセットを追加 |
| 4  | **0.666** | 学習率スケジューリングとbackboneアーキテクチャの実装修正 |
| 5  | 学習中 | 学習率スケジューリングやAugmentationの処理・パラメータを修正 |


## 所感

### 理解できた点

1. **conv4_3のL2Norm**
   - conv4_3は他の高次層と比べて出力のスケールが大きいため、バランスを合わせるためにL2Normが必要。
   - 実際に値を確認してみると、高次層の出力最大値は1~2の間に収まっていたが、L2Normありでもconv4_3は8程度になっていた。
   - 一番見落としそうな箇所な気がするが、これが違うだけで一気に性能が壊れそう。（未確認）

2. **Sample Patch、Zoom Out**
   - このAugmentationの有無で性能が大きく変化した。（特に小さい物体のAPが改善。）
   - 課題に対して適切なAugmentationを入れることで、これほど大きく性能が変わり得ると知れたのは学びだった。

3. **データセットの重要性**
   - データの質と量が正義であると改めて感じた。
   - AIモデルのアーキテクチャや学習方法に重きを置きがちだが、データの品質は最優先で気にするべき。

### 難しかった点

1. **論文準拠の実装**
   - 論文とどこが違うのかを見つけるのに苦労した。（生成AIと壁打ちしながらやっても40~50時間程度かかった。）
   - リポジトリ内に論文のPDFを置いて、生成AIに聞きながらやるようにしてからはスピードが上がった気がする。
   - 英語を読むのに慣れていないのも原因なので、色々な論文をひたすら読んでみるべし。

2. **ハイパーパラメータの調整**
   - 学習率や重み減衰などのパラメータの最適値がよくわからなかった。（今もよくわかっていない）
   - 学習に時間がかかりすぎるので、試行回数も限られるのが難しいと感じた。（実務では特に気にするところ）

3. **NMSの実装**
   - 実装自体はしたものの、処理が遅すぎて使い物にならなかった。
   - 結局ライブラリに頼った。便利なライブラリは使おう。


### 今後やりたいこと

1. **バックボーンの変更**: より効率的な軽量バックボーン（MobileNet, EfficientNet）での実験を深掘り
2. **最適化**: 処理の高速化、大きいモデルの蒸留・量子化など
