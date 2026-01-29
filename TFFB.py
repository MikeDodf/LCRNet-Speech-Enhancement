import torch
import torch.nn as nn
class ComplexTFFusionBlock(nn.Module):
    """
    复数时间-频率特征融合模块
    输入: (B, C, F, T, 2)
    输出: (B, C, F, T, 2)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 分别对实部和虚部分别处理
        self.freq_fc_real = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 聚合频率
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.freq_fc_imag = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.time_fc_real = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # 聚合时间
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.time_fc_imag = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 小型融合卷积
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, F, T, 2)
        """
        real, imag = x[..., 0], x[..., 1]  # 分开实部和虚部

        # 频率与时间注意力
        freq_real = self.freq_fc_real(real)
        time_real = self.time_fc_real(real)
        freq_imag = self.freq_fc_imag(imag)
        time_imag = self.time_fc_imag(imag)

        # 融合注意力
        real_out = real * freq_real * time_real
        imag_out = imag * freq_imag * time_imag

        # 融合卷积
        real_out = self.fusion(real_out)
        imag_out = self.fusion(imag_out)

        out = torch.stack([real_out, imag_out], dim=-1)
        return out
