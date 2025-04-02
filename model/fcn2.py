import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super().__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

class FCN(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        """
        Args:
            num_classes: 分割类别数
            pretrained_path: 自定义预训练权重路径
        """
        super().__init__()

        # 初始化不带预训练权重的Backbone
        backbone = resnet50(weights=None,
                            replace_stride_with_dilation=[False, True, True])

        # 加载自定义权重
        if pretrained_path:
            print(f"Loading custom pretrained weights from {pretrained_path}")
            backbone_state_dict = torch.load(pretrained_path)

            # 过滤不匹配的权重
            filtered_dict = {k: v for k, v in backbone_state_dict.items()
                             if k in backbone.state_dict() and
                             v.shape == backbone.state_dict()[k].shape}

            # 加载过滤后的权重
            backbone.load_state_dict(filtered_dict, strict=False)
            missing_keys = set(backbone_state_dict.keys()) - set(filtered_dict.keys())
            if missing_keys:
                print(f"Missing keys when loading backbone: {missing_keys}")

        # 中间层特征提取
        self.backbone = IntermediateLayerGetter(
            backbone,
            return_layers={'layer3': 'aux', 'layer4': 'out'}
        )

        # 分类头初始化
        self.classifier = FCNHead(2048, num_classes)
        self.aux_classifier = FCNHead(1024, num_classes)  # 辅助头

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        # 主分支处理
        main_out = self.classifier(features['out'])
        main_out = F.interpolate(
            main_out,
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )

        # 辅助分支处理
        if self.training:
            aux_out = self.aux_classifier(features['aux'])
            aux_out = F.interpolate(
                aux_out,
                size=input_shape,
                mode='bilinear',
                align_corners=False
            )
            return {'out': main_out, 'aux': aux_out}

        return main_out


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = FCN(
        num_classes=21,
        pretrained_path="../results/weights/fcn_resnet50_coco.pth"  # 自定义权重路径
    )

    # 验证前向传播
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output['out'].shape}")  # 应输出 torch.Size([2, 21, 512, 512])