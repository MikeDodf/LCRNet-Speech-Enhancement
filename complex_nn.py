import torch
import torch.nn as nn
import torch.nn.functional as F

class UniDeepFsmn(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        ## input: batch (b) x sequence(T) x feature (h)
        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = torch.unsqueeze(p1, 1)
        #x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        #x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)
        #out1: batch (b) x channel (c) x sequence(T) x feature (h)
        return input + out1.squeeze()

class ComplexUniDeepFsmn(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2]
        b,c,h,T,d = x.size()
        x = torch.reshape(x, (b, c*h, T, d))
        # x: [b,h,T,2]
        x = torch.transpose(x, 1, 2)
        # x: [b,T,h,2]
        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])        
        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)
        # output: [b,T,h,2]
        output = torch.stack((real, imaginary), dim=-1)
        # output: [b,h,T,2]
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (b, c, h, T, d))

        return output

class ComplexUniDeepFsmn_L1(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn_L1, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2]
        b,c,h,T,d = x.size()
        #x : [b,T,h,c,2]
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b*T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # output: [b*T,h,c,2]
        output = torch.stack((real, imaginary), dim=-1)
        output = torch.reshape(output, (b, T, h, c, d))
        # output: [b,c,h,T,2]
        output = torch.transpose(output, 1, 3)

        return output

class BidirectionalLSTM_L1(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_L1, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)

    def forward(self, input):
        output, _ = self.rnn(input)
        return output

class BidirectionalLSTM_L2(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_L2, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        #T: sequence length, b: batch, h: feature size
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ComplexBidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=1024):
        super(ComplexBidirectionalLSTM, self).__init__()

        self.lstm_re_L1 = BidirectionalLSTM_L1(nIn, nHidden, nOut)
        self.lstm_im_L1 = BidirectionalLSTM_L1(nIn, nHidden, nOut)
        self.lstm_re_L2 = BidirectionalLSTM_L2(nHidden, nHidden, nOut)
        self.lstm_im_L2 = BidirectionalLSTM_L2(nHidden, nHidden, nOut)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2]
        b,c,h,T,d = x.size()
        x = torch.reshape(x, (b, c*h, T, d))
        # x: [b,h,T,2]
        x = torch.transpose(x, 0, 2)
        # x: [T,h,b,2]
        x = torch.transpose(x, 1, 2)
        # x: [T,b,h,2]
        real_L1 = self.lstm_re_L1(x[..., 0]) - self.lstm_im_L1(x[..., 1])
        imaginary_L1 = self.lstm_re_L1(x[..., 1]) + self.lstm_im_L1(x[..., 0])
        real = self.lstm_re_L2(real_L1) - self.lstm_im_L2(imaginary_L1)
        imaginary = self.lstm_re_L2(imaginary_L1) + self.lstm_im_L2(real_L1)
        # output: [T,b,h,2]
        output = torch.stack((real, imaginary), dim=-1)
        # output: [T,h,b,2]
        output = torch.transpose(output, 1, 2) 
        # output: [b,h,T,2]
        output = torch.transpose(output, 0, 2)
        output = torch.reshape(output, (b, c, h, T, d))

        return output

class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

