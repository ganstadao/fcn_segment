import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

#直接调用
'''model = fcn_resnet50(
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    num_classes=21  
)'''

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
    def __init__(self, num_classes):
        super().__init__()
        # 使用 weights 参数加载预训练权重
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1,
                           replace_stride_with_dilation=[False, True, True])
        
        # 获取中间层输出
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
        
        # 主分支处理(上采样)
        main_out = self.classifier(features['out'])
        main_out = F.interpolate(
            main_out, 
            size=input_shape, 
            mode='bilinear', 
            align_corners=False  # 关键修改点
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
            return {'out' : main_out , 'aux' : aux_out}
        
        return main_out

# 验证代码
if __name__ == "__main__":
    # 参数设置
    num_classes = 21
    input_size = (3, 520,925)
    
    # 创建模型
    model = FCN(num_classes=num_classes)
    
    # 测试前向传播
    dummy_input = torch.randn(2, *input_size)
    
    # 训练模式
    model.train()
    outputs = model(dummy_input)
    print("Train mode outputs:")
    print(f"Main output shape: {outputs['out'].shape}")
    print(f"Aux output shape: {outputs['aux'].shape}")
    
    # 推理模式
    model.eval()
    output = model(dummy_input)
    print("\nEval mode output:")
    print(f"Output shape: {output.shape}")