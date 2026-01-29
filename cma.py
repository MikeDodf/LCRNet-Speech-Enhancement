import torch
from torch import nn


class EMAComplex(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMAComplex, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_f = nn.AdaptiveAvgPool2d((None, 1))  # 沿时间维度池化
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))  # 沿频率维度池化
        self.gn_r = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.gn_i = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1_r = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv1x1_i = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3_r = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv3x3_i = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, f, t, _ = x.size()
        # 分离实部和虚部
        x_r = x[:, :, :, :, 0]  # [B, C, F, T]
        x_i = x[:, :, :, :, 1]  # [B, C, F, T]

        # 分组处理
        group_x_r = x_r.reshape(b * self.groups, -1, f, t)  # [B*groups, C//groups, F, T]
        group_x_i = x_i.reshape(b * self.groups, -1, f, t)  # [B*groups, C//groups, F, T]

        # 多尺度池化
        x_f_r = self.pool_f(group_x_r)  # [B*groups, C//groups, F, 1]
        x_t_r = self.pool_t(group_x_r).permute(0, 1, 3, 2)  # [B*groups, C//groups, T, 1]
        x_f_i = self.pool_f(group_x_i)  # [B*groups, C//groups, F, 1]
        x_t_i = self.pool_t(group_x_i).permute(0, 1, 3, 2)  # [B*groups, C//groups, T, 1]

        # 融合频率和时间特征，引入交叉项
        hw_r = self.conv1x1_r(torch.cat([x_f_r, x_t_r], dim=2))  # [B*groups, C//groups, F+T, 1]
        hw_i = self.conv1x1_i(torch.cat([x_f_i, x_t_i], dim=2))  # [B*groups, C//groups, F+T, 1]
        x_f_r, x_t_r = torch.split(hw_r, [f, t], dim=2)  # [B*groups, C//groups, F, 1], [B*groups, C//groups, T, 1]
        x_f_i, x_t_i = torch.split(hw_i, [f, t], dim=2)  # [B*groups, C//groups, F, 1], [B*groups, C//groups, T, 1]

        # 交叉项：类似 SELayer 的实部和虚部交互
        x_f_r_weight = x_f_r.sigmoid() - x_f_i.sigmoid()  # 实部频率权重减去虚部影响
        x_f_i_weight = x_f_i.sigmoid() + x_f_r.sigmoid()  # 虚部频率权重加上实部影响
        x_t_r_weight = x_t_r.sigmoid() - x_t_i.sigmoid()  # 实部时间权重减去虚部影响
        x_t_i_weight = x_t_i.sigmoid() + x_t_r.sigmoid()  # 虚部时间权重加上实部影响

        # 应用空间注意力
        x1_r = self.gn_r(group_x_r * x_f_r_weight * x_t_r_weight.permute(0, 1, 3, 2))
        x1_i = self.gn_i(group_x_i * x_f_i_weight * x_t_i_weight.permute(0, 1, 3, 2))

        # 局部特征提取
        x2_r = self.conv3x3_r(group_x_r)
        x2_i = self.conv3x3_i(group_x_i)

        # 全局通道注意力
        x11_r = self.softmax(
            self.agp(x1_r).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # [B*groups, 1, C//groups]
        x12_r = x2_r.reshape(b * self.groups, c // self.groups, -1)  # [B*groups, C//groups, F*T]
        x21_r = self.softmax(self.agp(x2_r).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22_r = x1_r.reshape(b * self.groups, c // self.groups, -1)
        x11_i = self.softmax(self.agp(x1_i).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12_i = x2_i.reshape(b * self.groups, c // self.groups, -1)
        x21_i = self.softmax(self.agp(x2_i).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22_i = x1_i.reshape(b * self.groups, c // self.groups, -1)

        # 计算注意力权重
        weights_r = (torch.matmul(x11_r, x12_r) + torch.matmul(x21_r, x22_r)).reshape(b * self.groups, 1, f, t)
        weights_i = (torch.matmul(x11_i, x12_i) + torch.matmul(x21_i, x22_i)).reshape(b * self.groups, 1, f, t)

        # 应用权重并重塑输出
        out_r = (group_x_r * weights_r.sigmoid()).reshape(b, c, f, t)
        out_i = (group_x_i * weights_i.sigmoid()).reshape(b, c, f, t)
        return torch.stack([out_r, out_i], dim=4)  # [B, C, F, T, 2]


if __name__ == '__main__':
    block = EMAComplex(64).cuda()
    input = torch.rand(1, 64, 64, 64, 2).cuda()
    output = block(input)
    print(input.size(), output.size())