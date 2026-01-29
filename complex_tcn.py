import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """
    去掉卷积右侧多余的 padding，保证因果输出长度与输入长度一致。
    chomp_size = padding
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ComplexConv1d(nn.Module):
    """
    复数 1D 卷积（Depthwise + Pointwise）
    输入输出: (real, imag) 各 [B, C, T]
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Depthwise 卷积
        self.real_dw = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 padding=padding, dilation=dilation, groups=in_channels)
        self.imag_dw = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 padding=padding, dilation=dilation, groups=in_channels)

        # Pointwise 卷积
        self.real_pw = nn.Conv1d(in_channels, out_channels, 1)
        self.imag_pw = nn.Conv1d(in_channels, out_channels, 1)

        self.chomp = Chomp1d(padding)

    def forward(self, real, imag):
        # Depthwise
        r_dw = self.chomp(self.real_dw(real))
        i_dw = self.chomp(self.imag_dw(imag))

        # 升维
        r_dw = self.real_pw(r_dw)
        i_dw = self.imag_pw(i_dw)

        # Complex 组合
        real_out = r_dw - i_dw
        imag_out = r_dw + i_dw

        # 时序对齐
        real_out = self.chomp(real_out)
        imag_out = self.chomp(imag_out)
        return real_out, imag_out

class ComplexPReLU(nn.Module):
    """
    参数化 PReLU for complex, 提升非线性表达
    """
    def __init__(self, channels):
        super().__init__()
        self.prelu_r = nn.PReLU(channels)
        self.prelu_i = nn.PReLU(channels)

    def forward(self, real, imag):
        return self.prelu_r(real), self.prelu_i(imag)


class TemporalAttention(nn.Module):
    """
    轻量 Causal Temporal Attention (SE-like), 提升时间依赖捕捉
    """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, real, imag):
        real_att = self.fc(self.avg_pool(real).squeeze(-1)).unsqueeze(-1)
        imag_att = self.fc(self.avg_pool(imag).squeeze(-1)).unsqueeze(-1)
        return real * real_att, imag * imag_att


class ComplexDropout(nn.Module):
    def __init__(self, p=0.0):  # Set to 0 for efficiency
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, real, imag):
        return self.drop(real), self.drop(imag)


class ComplexTemporalBlock(nn.Module):
    """
    一个 Complex TCN Block: 两层 ComplexConv1d + PReLU + attention + dropout + 残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = ComplexConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.prelu1 = ComplexPReLU(out_channels)
        self.drop1 = ComplexDropout(dropout)
        self.att1 = TemporalAttention(out_channels)

        self.conv2 = ComplexConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.prelu2 = ComplexPReLU(out_channels)
        self.drop2 = ComplexDropout(dropout)
        self.att2 = TemporalAttention(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, real, imag):
        # 主分支
        r1, i1 = self.conv1(real, imag)
        r1, i1 = self.prelu1(r1, i1)
        r1, i1 = self.drop1(r1, i1)
        r1, i1 = self.att1(r1, i1)

        r2, i2 = self.conv2(r1, i1)
        r2, i2 = self.prelu2(r2, i2)
        r2, i2 = self.drop2(r2, i2)
        r2, i2 = self.att2(r2, i2)

        # 残差
        if self.downsample is not None:
            res_r = self.downsample(real)
            res_i = self.downsample(imag)
        else:
            res_r, res_i = real, imag

        min_len = min(r2.size(-1), res_r.size(-1))
        r2 = r2[..., :min_len]
        i2 = i2[..., :min_len]
        res_r = res_r[..., :min_len]
        res_i = res_i[..., :min_len]

        out_r = self.final_relu(r2 + res_r)
        out_i = self.final_relu(i2 + res_i)
        return out_r, out_i


class ComplexTCN(nn.Module):
    """
    轻量 ComplexTCN: 2 blocks, dilation=1 for CPU
    """
    def __init__(self, in_channels, channel_list=[32, 64], kernel_size=2, dropout=0.0):
        super().__init__()
        blocks = []
        for i, out_ch in enumerate(channel_list):
            dilation = 1  # Fixed to 1 for low computation
            in_ch = in_channels if i == 0 else channel_list[i - 1]
            blocks.append(ComplexTemporalBlock(in_ch, out_ch, kernel_size=kernel_size,
                                               dilation=dilation, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        assert x.dim() == 5 and x.size(-1) == 2, "Input must be (B,C,D,T,2)"
        B, C, D, T, _ = x.shape

        real = x[..., 0]
        imag = x[..., 1]

        real = real.permute(0, 2, 1, 3).contiguous().view(B * D, C, T)
        imag = imag.permute(0, 2, 1, 3).contiguous().view(B * D, C, T)

        for block in self.blocks:
            real, imag = block(real, imag)

        C_out = real.size(1)
        real = real.view(B, D, C_out, -1).permute(0, 2, 1, 3).contiguous()
        imag = imag.view(B, D, C_out, -1).permute(0, 2, 1, 3).contiguous()

        out = torch.stack([real, imag], dim=-1)
        return out


# ----------------- TEST -----------------
if __name__ == "__main__":
    # 随机测试，确保前向不报错且 shape 正确
    B, C, D, T = 2, 64, 257, 100
    x = torch.randn(B, C, D, T, 2)

    model = ComplexTCN(in_channels=C, channel_list=[32, 64], kernel_size=2, dropout=0.0)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)  # 期望 [B, C_out, D, T, 2]