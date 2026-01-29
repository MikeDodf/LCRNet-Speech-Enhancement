import torch
import torch.nn as nn

class StreamingComplexFSMN(nn.Module):
    """
    流式复数 FSMN 模块，用于时序特征建模。
    支持逐帧更新记忆，可在实时系统中使用。
    """
    def __init__(self, dim, memory_size=10):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        # 复数记忆滤波器参数
        self.real_weight = nn.Parameter(torch.randn(memory_size))
        self.imag_weight = nn.Parameter(torch.randn(memory_size))

        # 流式记忆缓存（初始化为空）
        self.memory_real = None
        self.memory_imag = None

    def reset_state(self, batch_size, device):
        """重置记忆状态"""
        self.memory_real = torch.zeros(batch_size, self.memory_size, self.dim, device=device)
        self.memory_imag = torch.zeros(batch_size, self.memory_size, self.dim, device=device)

    def forward(self, real, imag):
        """
        real, imag: [B, T, D]
        """
        B, T, D = real.shape
        out_real = torch.zeros_like(real)
        out_imag = torch.zeros_like(imag)

        if self.memory_real is None:
            self.reset_state(B, real.device)

        for t in range(T):
            r_t, i_t = real[:, t, :], imag[:, t, :]

            # 计算历史帧加权和
            mem_r = torch.sum(self.real_weight.view(1, -1, 1) * self.memory_real, dim=1)
            mem_i = torch.sum(self.imag_weight.view(1, -1, 1) * self.memory_imag, dim=1)

            # 复数加法
            out_real[:, t, :] = r_t + mem_r
            out_imag[:, t, :] = i_t + mem_i

            # 更新记忆窗口（FIFO）
            self.memory_real = torch.cat([r_t.unsqueeze(1), self.memory_real[:, :-1, :]], dim=1)
            self.memory_imag = torch.cat([i_t.unsqueeze(1), self.memory_imag[:, :-1, :]], dim=1)

        return out_real, out_imag