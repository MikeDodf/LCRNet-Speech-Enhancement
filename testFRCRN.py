import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np

# 假设 ConvSTFT、ConviSTFT 和 UNet 已正确定义
sys.path.append(os.path.dirname(__file__))

# 导入 DCCRN 模型代码（假设与测试代码在同一文件或模块中）
from frcrn import FRCRN_Wrapper_StandAlone, FRCRN_SE_16K, DCCRN

def test_dccrn():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义超参数
    args = argparse.Namespace(
        win_len=640,
        win_inc=320,
        fft_len=640,
        win_type='hanning'
    )

    # 初始化模型
    try:
        model_standalone = FRCRN_Wrapper_StandAlone(args).to(device)
        model_se_16k = FRCRN_SE_16K(args).to(device)
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # 模拟输入数据（16kHz，2秒音频，批次大小=2）
    batch_size = 2
    sample_rate = 16000
    audio_length = sample_rate * 2
    noisy_wav = torch.randn(batch_size, 1, audio_length).to(device)  # 带噪音频
    clean_wav = torch.randn(batch_size, 1, audio_length).to(device)  # 干净音频

    # 测试 FRCRN_Wrapper_StandAlone 前向传播
    try:
        model_standalone.eval()
        with torch.no_grad():
            output_standalone = model_standalone(noisy_wav)
            print(f"FRCRN_Wrapper_StandAlone 输出形状: {output_standalone.shape}")
            assert output_standalone.shape == (batch_size, audio_length), "Standalone 输出形状错误"
    except Exception as e:
        print(f"FRCRN_Wrapper_StandAlone 前向传播失败: {e}")
        return

    # 测试 FRCRN_SE_16K 前向传播
    try:
        model_se_16k.eval()
        with torch.no_grad():
            output_se_16k = model_se_16k(noisy_wav)
            print(f"FRCRN_SE_16K 输出形状: {output_se_16k.shape}")
            assert output_se_16k.shape == (batch_size, audio_length), "SE_16K 输出形状错误"
    except Exception as e:
        print(f"FRCRN_SE_16K 前向传播失败: {e}")
        return

    # 测试 DCCRN 模型的核心功能
    dccrn_model = DCCRN(
        complex=True,
        model_complexity=45,
        model_depth=14,
        log_amp=False,
        padding_mode="zeros",
        win_len=args.win_len,
        win_inc=args.win_inc,
        fft_len=args.fft_len,
        win_type=args.win_type
    ).to(device)

    # 测试前向传播
    try:
        dccrn_model.eval()
        with torch.no_grad():
            out_list = dccrn_model(noisy_wav)
            est_spec, est_wav, est_mask = out_list
            print(f"DCCRN 输出形状 - est_spec: {est_spec.shape}, est_wav: {est_wav.shape}, est_mask: {est_mask.shape}")
            expected_spec_shape = (batch_size, args.fft_len // 2 + 1 + args.fft_len // 2 + 1, audio_length // args.win_inc)
            expected_wav_shape = (batch_size, audio_length)
            expected_mask_shape = (batch_size, args.fft_len // 2 + 1 + args.fft_len // 2 + 1, audio_length // args.win_inc)
            assert est_spec.shape == expected_spec_shape, "估计频谱形状错误"
            assert est_wav.shape == expected_wav_shape, "估计波形形状错误"
            assert est_mask.shape == expected_mask_shape, "估计掩码形状错误"
    except Exception as e:
        print(f"DCCRN 前向传播失败: {e}")
        return

    # 测试推理模式
    try:
        with torch.no_grad():
            est_wav_inference = dccrn_model.inference(noisy_wav)
            print(f"DCCRN 推理输出形状: {est_wav_inference.shape}")
            assert est_wav_inference.shape == (audio_length,), "推理输出形状错误"
    except Exception as e:
        print(f"DCCRN 推理失败: {e}")
        return

    # 测试损失函数
    try:
        # Si-SNR 损失
        loss_sisnr = dccrn_model.loss(noisy_wav, clean_wav, out_list, device, mode='SiSNR')
        print(f"Si-SNR 损失: {loss_sisnr.item()}")
        assert isinstance(loss_sisnr, torch.Tensor) and loss_sisnr.dim() == 0, "Si-SNR 损失格式错误"

        # 混合损失
        loss_mix, mask_real_loss, mask_imag_loss = dccrn_model.loss(noisy_wav, clean_wav, out_list, device, mode='Mix')
        print(f"混合损失: {loss_mix.item()}, 实部掩码损失: {mask_real_loss.item()}, 虚部掩码损失: {mask_imag_loss.item()}")
        assert isinstance(loss_mix, torch.Tensor) and loss_mix.dim() == 0, "混合损失格式错误"
        assert isinstance(mask_real_loss, torch.Tensor) and mask_real_loss.dim() == 0, "实部掩码损失格式错误"
        assert isinstance(mask_imag_loss, torch.Tensor) and mask_imag_loss.dim() == 0, "虚部掩码损失格式错误"
    except Exception as e:
        print(f"损失计算失败: {e}")
        return

    print("所有测试通过！模型运行正常。")

if __name__ == "__main__":
    test_dccrn()