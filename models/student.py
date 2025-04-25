import timm
import torch.nn as nn


class TinyViTStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(
            config.student['arch'],
            pretrained=config.student['pretrained'],
            features_only=True,
            embed_dims=config.student['embed_dims'],
            depths=config.student['depths'],
            num_heads=config.student['num_heads'],
            window_sizes=config.student['window_sizes']
        )
        self.head = nn.Linear(
            config.student['embed_dims'][-1],
            config.data['num_classes']
        )

        # 特征对齐层
        self.align_layers = nn.ModuleList([
            nn.Conv2d(dim, t_dim, 1)
            for dim, t_dim in zip(
                config.student['embed_dims'][::2],  # 选取stage1和stage3
                config.teacher['feature_dims']
            )
        ])

    def forward(self, x):
        features = self.backbone(x)
        return {
            'features': [self.align_layers[0](features[1]),
                         self.align_layers[1](features[3])],
            'logits': self.head(features[-1].mean(dim=[2, 3]))
        }