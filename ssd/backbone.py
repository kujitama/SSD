import torch
import torch.nn as nn
import timm

def build_backbone(cfg):
    """timmを使ったバックボーンネットワークの構築"""
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
    
    
class VGG16Backbone(nn.Module):
    def __init__(self, backbone, backbone_name):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        
        # MaxPool4をceil_mode=Trueに変更
        # これをしないと特徴マップのサイズが小さくなるため、headで出力するボックスも減ってしまう
        backbone.features[16] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        
        # VGG16のfeaturesから必要な層を抽出
        # conv4_3: features[22] (ReLU4_3の出力) - 38x38, 512ch
        # conv5_3: features[29] (ReLU5_3の出力) - 19x19, 512ch
        self.conv4_3 = backbone.features[:23]  # conv4_3まで (index 22のReLUまで含む)
        self.conv5_3 = backbone.features[23:30]  # conv4_3の次からconv5_3まで
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # SSD用の追加畳み込み層 
        self.extra_layers = nn.ModuleList([
            # conv8_1, conv8_2
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            # conv9_1, conv9_2
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            # conv10_1, conv10_2
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
            # conv11_1, conv11_2
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        ])
        
        # He初期化
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for m in self.conv6.modules():
            weights_init(m)
        for m in self.conv7.modules():
            weights_init(m)
        for m in self.extra_layers.modules():
            weights_init(m)
    
    def forward(self, x):
        features = []
        
        # conv4_3 特徴量 (38x38, 512ch)
        x = self.conv4_3(x)
        features.append(x)
        
        # conv5_3 -> conv6 -> conv7 特徴量 (19x19, 1024ch)
        x = self.conv5_3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        features.append(x)
        
        # 追加の特徴量レイヤーを適用
        for extra_layer in self.extra_layers:
            x = extra_layer(x)
            features.append(x)
        
        # 最終的に6つの特徴マップ: [38x38, 19x19, 10x10, 5x5, 3x3, 1x1]
        return features
