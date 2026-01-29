import torch
import os
import argparse
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper
from utils.misc import reload_for_eval
from utils.decode import decode_one_audio

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def process_single_audio(input_path, output_path, config_path='config/inference/FRCRN_SE_16K.yaml'):
    # 解析配置文件和参数
    parser = yamlargparse.ArgumentParser("Settings")
    parser.add_argument('--config', help='config file path', default=config_path, action=yamlargparse.ActionConfigFile)
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN',
                        help='the checkpoint dir')
    parser.add_argument('--use-cuda', dest='use_cuda', default=0, type=int, help='use cuda')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000)
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    print('Creating model...')
    model = network_wrapper(args).se_network
    model.to(device)

    # 加载模型
    print('Loading model...')
    try:
        reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    with torch.no_grad():
        # 读取输入音频
        print(f"Processing audio: {input_path}")
        try:
            # 读取音频文件
            input_audio, sr = sf.read(input_path)
            if sr != args.sampling_rate:
                raise ValueError(f"Input audio sampling rate {sr} does not match required {args.sampling_rate}")

            # 转换为张量并调整维度（假设模型需要 [1, T] 格式）
            input_audio = torch.from_numpy(input_audio).float().unsqueeze(0).to(device)
            input_len = input_audio.shape[-1]
            scalar = 1.0  # 假设没有额外的标量调整，视 DataReader 实现而定

            # 处理音频
            output_audio = decode_one_audio(model, device, input_audio, args)
            output_audio = output_audio[:input_len] * scalar

            # 保存输出音频
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            sf.write(output_path, output_audio.cpu().numpy(), args.sampling_rate)
            print(f"Saved enhanced audio to: {output_path}")

        except Exception as e:
            print(f"Error processing audio: {e}")
            return

    print('Done!')


if __name__ == "__main__":
    # 示例：处理单个音频文件
    input_audio_path = 'data/input.wav'  # 替换为你的输入音频路径
    output_audio_path = 'outputs/FRCRN_SE_16K/enhanced_output.wav'  # 替换为你的输出音频路径
    process_single_audio(input_audio_path, output_audio_path)