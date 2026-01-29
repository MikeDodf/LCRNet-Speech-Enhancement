import torch
import torch.nn as nn
import torch.nn.functional as F
import complex_nn as complex_nn  # 假设已定义 ComplexConv2d 等

from cma import EMAComplex  # CMA 模块
from TFFB import ComplexTFFusionBlock  # TFFB 模块
from steamFSMN import StreamingComplexFSMN  # FSMN 模块


# 编码器模块，用于下采样提取高级特征
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False,
                 padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' 填充
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 解码器模块，用于上采样重建特征
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), output_padding=(0, 0),
                 complex=False):
        super().__init__()
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 优化的 UNet，深度 8，添加 EMAComplex (CMA) 注意力机制、ComplexTFFusionBlock (TFFB) 和 StreamingComplexFSMN (FSMN)，并使用跳跃连接
class UNet(nn.Module):
    def __init__(self, input_channels=1, complex=False, model_complexity=45, model_depth=8, padding_mode="zeros", fsmn_memory=10):
        super().__init__()
        if complex:
            model_complexity = int(model_complexity // 1.414)  # 复数模式下缩放复杂度

        self.complex = complex
        self.padding_mode = padding_mode
        self.model_depth = model_depth
        self.model_length = model_depth // 2

        # 计算缩放因子
        self.scale_factor = model_complexity / 45.0

        # 辅助函数：确保通道数能被8整除
        def make_divisible(v, divisor=8):
            """确保通道数能被divisor整除"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 设置基础通道配置
        self.set_size(model_depth=model_depth, input_channels=input_channels)

        # 缩放后的实际通道数（确保能被8整除）
        self.scaled_enc_channels = [input_channels] + [
            make_divisible(ch * self.scale_factor) for ch in self.enc_channels[1:]
        ]

        # 构建编码器
        self.encoders = nn.ModuleList()
        self.cma_layers_enc = nn.ModuleList()

        for i in range(self.model_length):
            in_ch = self.scaled_enc_channels[i]
            out_ch = self.scaled_enc_channels[i + 1]
            module = Encoder(in_ch, out_ch, kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i],
                             complex=complex, padding_mode=padding_mode)
            self.encoders.append(module)
            # 每个编码器后添加CMA注意力
            factor = self._get_valid_factor(out_ch)
            cma_layer_enc = EMAComplex(out_ch, factor=factor)
            self.cma_layers_enc.append(cma_layer_enc)

        # 在编码器最后一层后添加 TFFB 和 FSMN
        bottleneck_ch = self.scaled_enc_channels[-1]
        self.tffb = ComplexTFFusionBlock(channels=bottleneck_ch)
        self.fsmn = StreamingComplexFSMN(dim=bottleneck_ch, memory_size=fsmn_memory)

        # 构建解码器和融合层
        self.decoders = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()  # 用于融合跳跃连接的1x1卷积
        self.cma_layers_dec = nn.ModuleList()

        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d

        for i in range(self.model_length):
            # 第一个解码器输入是瓶颈层
            if i == 0:
                in_ch = self.scaled_enc_channels[-1]
                out_ch = self.scaled_enc_channels[-2]
            else:
                # 后续解码器：输入是融合后的特征
                in_ch = self.scaled_enc_channels[self.model_length - i]
                out_ch = self.scaled_enc_channels[self.model_length - i - 1]

            # 解码器
            module = Decoder(in_ch, out_ch, kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i],
                             output_padding=self.dec_output_paddings[i], complex=complex)
            self.decoders.append(module)

            # 前 model_length-1 个解码器需要CMA和融合层
            if i < self.model_length - 1:
                # CMA层
                factor = self._get_valid_factor(out_ch)
                cma_layer_dec = EMAComplex(out_ch, factor=factor)
                self.cma_layers_dec.append(cma_layer_dec)

                # 融合层：将解码器输出和跳跃连接融合
                # 输入通道：out_ch (解码器输出) + out_ch (跳跃连接) = 2*out_ch
                # 输出通道：out_ch (保持维度)
                fuse_conv = conv(2 * out_ch, out_ch, kernel_size=1, stride=1, padding=0)
                self.fuse_convs.append(fuse_conv)

        # 最终输出层
        final_ch = self.scaled_enc_channels[0]  # 应该等于input_channels
        self.linear = conv(final_ch, 1, kernel_size=1)

    def _get_valid_factor(self, channels):
        """获取有效的分组因子"""
        factor = min(8, channels)
        while channels % factor != 0 and factor > 1:
            factor -= 1
        return factor

    def forward(self, inputs):
        # 保存输入维度
        input_shape = inputs.shape
        batch_size, channels, freq_dim, time_dim, complex_dim = input_shape

        # 编码阶段
        x = inputs
        enc_features = []  # 存储编码器输出

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            x = self.cma_layers_enc[i](x)  # 应用CMA注意力
            enc_features.append(x)

        # 瓶颈：应用 TFFB
        x = self.tffb(x)

        # 应用 FSMN：需要重塑为时间序列
        F_down = x.shape[2]
        T_enc = x.shape[3]
        C_enc = x.shape[1]  # 通道在 dim=1

        # 提取时间序列维度：重塑为每频率 bin 的时间序列 [B * F_down, T_enc, C_enc]
        real = x[..., 0].permute(0, 2, 3, 1).contiguous().view(batch_size * F_down, T_enc, C_enc)
        imag = x[..., 1].permute(0, 2, 3, 1).contiguous().view(batch_size * F_down, T_enc, C_enc)

        # 重置 FSMN 状态以匹配当前批次大小 batch_size * F_down
        self.fsmn.reset_state(batch_size=batch_size * F_down, device=real.device)

        # FSMN 流式建模
        real_out, imag_out = self.fsmn(real, imag)

        # 恢复形状 [batch_size * F_down, T_enc, C_enc] -> [batch_size, F_down, T_enc, C_enc, 2] -> [batch_size, C_enc, F_down, T_enc, 2]
        x = torch.stack([real_out, imag_out], dim=-1)
        x = x.view(batch_size, F_down, T_enc, C_enc, complex_dim).permute(0, 3, 1, 2, 4)

        # 解码阶段
        p = x  # 从瓶颈层开始

        for i, decoder in enumerate(self.decoders):
            # 上采样
            p = decoder(p)

            # 前 model_length-1 层需要CMA和跳跃连接
            if i < self.model_length - 1:
                # 应用CMA
                p = self.cma_layers_dec[i](p)

                # 获取对应的跳跃连接
                skip_idx = self.model_length - 2 - i
                skip = enc_features[skip_idx]

                # 确保空间维度匹配
                if skip.shape[2] != p.shape[2] or skip.shape[3] != p.shape[3]:
                    # 使用插值调整skip的尺寸
                    if self.complex:
                        # 复数张量：分别处理实部和虚部
                        skip_r = F.interpolate(skip[..., 0], size=(p.shape[2], p.shape[3]),
                                               mode='bilinear', align_corners=False)
                        skip_i = F.interpolate(skip[..., 1], size=(p.shape[2], p.shape[3]),
                                               mode='bilinear', align_corners=False)
                        skip = torch.stack([skip_r, skip_i], dim=-1)
                    else:
                        skip = F.interpolate(skip, size=(p.shape[2], p.shape[3]),
                                             mode='bilinear', align_corners=False)

                # 拼接并融合
                p_cat = torch.cat([p, skip], dim=1)
                p = self.fuse_convs[i](p_cat)

        # 最终输出
        cmp_spec = self.linear(p)

        # 确保输出尺寸与输入一致
        if cmp_spec.shape[2] != freq_dim or cmp_spec.shape[3] != time_dim:
            if self.complex:
                # 复数张量
                out_r = F.interpolate(cmp_spec[..., 0], size=(freq_dim, time_dim),
                                      mode='bilinear', align_corners=False)
                out_i = F.interpolate(cmp_spec[..., 1], size=(freq_dim, time_dim),
                                      mode='bilinear', align_corners=False)
                cmp_spec = torch.stack([out_r, out_i], dim=-1)
            else:
                cmp_spec = F.interpolate(cmp_spec, size=(freq_dim, time_dim),
                                         mode='bilinear', align_corners=False)

        # 确保输出形状正确
        cmp_spec = cmp_spec.contiguous()
        if self.complex:
            cmp_spec = cmp_spec.view(batch_size, 1, freq_dim, time_dim, complex_dim)

        return cmp_spec

    def set_size(self, model_depth=8, input_channels=1):
        """设置基础通道配置（未缩放）"""
        if model_depth == 8:
            # 编码器通道（从输入到瓶颈）
            self.enc_channels = [input_channels, 64, 64, 128, 128]
            self.enc_kernel_sizes = [(5, 2), (5, 2), (5, 2), (5, 2)]
            self.enc_strides = [(2, 1), (2, 1), (2, 1), (2, 1)]
            self.enc_paddings = [(2, 1), (2, 1), (2, 1), (2, 1)]

            # 解码器配置（对应的上采样）
            self.dec_kernel_sizes = [(5, 2), (5, 2), (5, 2), (5, 2)]
            self.dec_strides = [(2, 1), (2, 1), (2, 1), (2, 1)]
            self.dec_paddings = [(2, 1), (2, 1), (2, 1), (2, 1)]
            self.dec_output_paddings = [(0, 0), (0, 0), (0, 0), (0, 0)]

        elif model_depth == 6:
            self.enc_channels = [input_channels, 32, 64, 128]
            self.enc_kernel_sizes = [(3, 2), (3, 2), (3, 2)]
            self.enc_strides = [(4, 1), (4, 1), (2, 1)]
            self.enc_paddings = [(1, 1), (1, 1), (1, 1)]

            self.dec_kernel_sizes = [(3, 2), (3, 2), (3, 2)]
            self.dec_strides = [(2, 1), (4, 1), (4, 1)]
            self.dec_paddings = [(1, 1), (1, 1), (1, 1)]
            self.dec_output_paddings = [(0, 0), (0, 0), (0, 0)]

        elif model_depth == 4:
            self.enc_channels = [input_channels, 32, 64]
            self.enc_kernel_sizes = [(3, 2), (3, 2)]
            self.enc_strides = [(8, 1), (5, 1)]
            self.enc_paddings = [(1, 1), (1, 1)]

            self.dec_kernel_sizes = [(3, 2), (3, 2)]
            self.dec_strides = [(5, 1), (8, 1)]
            self.dec_paddings = [(1, 1), (1, 1)]
            self.dec_output_paddings = [(0, 0), (0, 0)]
        else:
            raise ValueError(f"未知的模型深度: {model_depth}")


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 UNet with CMA, TFFB, and FSMN")
    print("=" * 60)

    # 创建模型
    model = UNet(input_channels=1, complex=True, model_complexity=45, model_depth=8)
    model.eval()

    # 测试不同的输入尺寸
    test_cases = [
        (2, 1, 256, 100, 2),
        (1, 1, 257, 64, 2),
        (4, 1, 512, 128, 2),
    ]

    for batch_size, channels, freq_dim, time_dim, complex_dim in test_cases:
        print(f"\n测试输入形状: ({batch_size}, {channels}, {freq_dim}, {time_dim}, {complex_dim})")
        test_input = torch.randn(batch_size, channels, freq_dim, time_dim, complex_dim)

        with torch.no_grad():
            output = model(test_input)

        print(f"输出形状: {output.shape}")
        print(f"形状匹配: {output.shape == test_input.shape}")

        if output.shape != test_input.shape:
            print(f"⚠️  警告：输出形状不匹配！")
        else:
            print(f"✓ 通过")

    # 打印模型信息
    print("\n" + "=" * 60)
    print("模型信息")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 打印架构摘要
    print("\n编码器通道数:", model.scaled_enc_channels)
    print("编码器层数:", len(model.encoders))
    print("解码器层数:", len(model.decoders))
    print("CMA编码器层数:", len(model.cma_layers_enc))
    print("CMA解码器层数:", len(model.cma_layers_dec))
    print("融合层数:", len(model.fuse_convs))