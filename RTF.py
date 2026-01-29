import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(sys.path[0]))

from utils.misc import reload_for_eval
from utils.decode import decode_one_audio
from dataloder.dataloader import DataReader
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper
import numpy as np
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


def inference_with_rtf(args):
    device = torch.device('cuda') if args.use_cuda == 1 else torch.device('cpu')
    print(f"Device: {device}")
    print('Creating model...')
    model = network_wrapper(args).se_network
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal number of model parameters: {total_params:,}\n")

    print('Loading model...')
    reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
    model.eval()

    # RTF统计变量
    rtf_list = []
    total_audio_duration = 0.0
    total_processing_time = 0.0

    with torch.no_grad():
        data_reader = DataReader(args)
        output_wave_dir = args.output_dir
        print(f"Output directory: {output_wave_dir}")

        if not os.path.isdir(output_wave_dir):
            os.makedirs(output_wave_dir)

        num_samples = len(data_reader)
        print(f'Processing {num_samples} audio files...\n')
        print('-' * 80)
        print(f"{'File':<30} {'Duration(s)':<12} {'Process(s)':<12} {'RTF':<10}")
        print('-' * 80)

        for idx in range(num_samples):
            input_audio, wav_id, input_len, scalar = data_reader[idx]

            # 计算音频时长（秒）
            audio_duration = input_len / args.sampling_rate

            # 预热GPU（首次推理可能较慢）
            if idx == 0 and args.use_cuda:
                _ = decode_one_audio(model, device, input_audio, args)
                torch.cuda.synchronize()

            # 开始计时
            start_time = time.time()

            # 推理
            output_audio = decode_one_audio(model, device, input_audio, args)

            # GPU同步（确保计时准确）
            if args.use_cuda:
                torch.cuda.synchronize()

            # 结束计时
            end_time = time.time()
            processing_time = end_time - start_time

            # 计算RTF
            rtf = processing_time / audio_duration

            # 记录统计数据
            rtf_list.append(rtf)
            total_audio_duration += audio_duration
            total_processing_time += processing_time

            # 恢复音频幅度
            output_audio = output_audio[:input_len] * scalar

            # 保存音频
            new_path = os.path.basename(wav_id)
            output_path = os.path.join(output_wave_dir, new_path)
            sf.write(output_path, output_audio, args.sampling_rate)

            # 打印单个文件的RTF
            print(f"{new_path:<30} {audio_duration:<12.3f} {processing_time:<12.4f} {rtf:<10.4f}")

        print('-' * 80)
        print('\n' + '=' * 80)
        print('RTF Statistics Summary')
        print('=' * 80)

        # 计算统计指标
        rtf_array = np.array(rtf_list)
        mean_rtf = np.mean(rtf_array)
        median_rtf = np.median(rtf_array)
        std_rtf = np.std(rtf_array)
        min_rtf = np.min(rtf_array)
        max_rtf = np.max(rtf_array)
        overall_rtf = total_processing_time / total_audio_duration

        print(f"Total files processed    : {num_samples}")
        print(f"Total audio duration     : {total_audio_duration:.2f} seconds")
        print(f"Total processing time    : {total_processing_time:.2f} seconds")
        print(f"-" * 80)
        print(f"Overall RTF              : {overall_rtf:.4f}")
        print(f"Mean RTF                 : {mean_rtf:.4f}")
        print(f"Median RTF               : {median_rtf:.4f}")
        print(f"Std RTF                  : {std_rtf:.4f}")
        print(f"Min RTF                  : {min_rtf:.4f}")
        print(f"Max RTF                  : {max_rtf:.4f}")
        print(f"-" * 80)

        # 实时性能评估
        if overall_rtf < 1.0:
            speedup = 1.0 / overall_rtf
            print(f"✓ Real-time capable      : Yes ({speedup:.2f}x faster than real-time)")
        else:
            print(f"✗ Real-time capable      : No ({overall_rtf:.2f}x slower than real-time)")

        print(f"Model parameters         : {total_params:,}")
        print('=' * 80)

        # 保存RTF结果到文件
        rtf_log_path = os.path.join(output_wave_dir, 'rtf_results.txt')
        with open(rtf_log_path, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('RTF Test Results\n')
            f.write('=' * 80 + '\n\n')
            f.write(f"Model: {args.network}\n")
            f.write(f"Checkpoint: {args.checkpoint_dir}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Total parameters: {total_params:,}\n\n")
            f.write(f"Total files processed    : {num_samples}\n")
            f.write(f"Total audio duration     : {total_audio_duration:.2f} seconds\n")
            f.write(f"Total processing time    : {total_processing_time:.2f} seconds\n\n")
            f.write(f"Overall RTF              : {overall_rtf:.4f}\n")
            f.write(f"Mean RTF                 : {mean_rtf:.4f}\n")
            f.write(f"Median RTF               : {median_rtf:.4f}\n")
            f.write(f"Std RTF                  : {std_rtf:.4f}\n")
            f.write(f"Min RTF                  : {min_rtf:.4f}\n")
            f.write(f"Max RTF                  : {max_rtf:.4f}\n\n")

            f.write('-' * 80 + '\n')
            f.write('Per-file RTF Details\n')
            f.write('-' * 80 + '\n')
            f.write(f"{'File':<40} {'Duration(s)':<15} {'Process(s)':<15} {'RTF':<10}\n")
            f.write('-' * 80 + '\n')

            for idx in range(num_samples):
                input_audio, wav_id, input_len, scalar = data_reader[idx]
                new_path = os.path.basename(wav_id)
                audio_duration = input_len / args.sampling_rate
                f.write(
                    f"{new_path:<40} {audio_duration:<15.3f} {total_processing_time / num_samples:<15.4f} {rtf_list[idx]:<10.4f}\n")

        print(f"\nRTF results saved to: {rtf_log_path}")

    print('\nDone!')


if __name__ == "__main__":
    parser = yamlargparse.ArgumentParser("RTF Testing Settings")
    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile)

    parser.add_argument('--mode', type=str, default='inference', help='run train or inference')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/LCRNet',
                        help='the checkpoint dir')
    parser.add_argument('--input-path', dest='input_path', type=str, default='data/noise1.scp',
                        help='input dir or scp file for saving noisy audio')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='outputs/DNS_RTF',
                        help='output dir for saving processed audio')
    parser.add_argument('--use-cuda', dest='use_cuda', default=0, type=int, help='use cuda')
    parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='the num gpus to use')
    parser.add_argument('--network', type=str, default='FRCRN_SE_16K', help='select speech enhancement models')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000)
    parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=int, default=120,
                        help='the max length (second) for one-time pass decoding')
    parser.add_argument('--decode-window', dest='decode_window', type=int, default=4,
                        help='segmental decoding window length (second)')

    ## FFT Parameters
    parser.add_argument('--window-len', dest='win_len', type=int, default=640, help='the window-len in enframe')
    parser.add_argument('--window-inc', dest='win_inc', type=int, default=320, help='the window include in enframe')
    parser.add_argument('--fft-len', dest='fft_len', type=int, default=640,
                        help='the fft length when in extract feature')
    parser.add_argument('--num-mels', dest='num_mels', type=int, default=60,
                        help='the number of mels used for mossformer')
    parser.add_argument('--window-type', dest='win_type', type=str, default='hamming',
                        help='the window type in enframe')

    args = parser.parse_args()
    print(args)

    # 运行RTF测试
    inference_with_rtf(args)