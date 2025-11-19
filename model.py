# Model.py
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
from channel import Channel


""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """


def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32,
                                    kernel_size=5, padding=2)  # padding size could be changed here
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            # Simple power normalization: ||z||^2 = P * N
            batch_size = z_hat.size(0)
            N = z_hat[0].numel()
            power = torch.mean(z_hat ** 2, dim=[1,2,3], keepdim=True)  # [B,1,1,1]
            z_norm = z_hat * torch.sqrt(P / (power + 1e-8))
            return z_norm
        return _inner

# 在 _Encoder 类的 forward 函数最后（替换原来的 return x）
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            # DO NOT apply norm here — norm is only for analog channels
            # So just return raw features
            return x
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        # x = self.imgae_normalization(x)
        return x


# 在 model.py 中替换 DeepJSCC 类
class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=None, ber=0.01):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.decoder = _Decoder(c=c)
        self.channel_type = channel_type
        if channel_type == 'BSC':
            self.ber = ber
            self.channel = None  # handled in forward
        else:
            if snr is not None:
                self.channel = Channel(channel_type, snr)
            else:
                self.channel = None

    def forward(self, x):
        z = self.encoder(x)  # [B, 2c, H, W]

        if self.channel_type == 'BSC':
            # Flatten and apply sigmoid + STE
            z_flat = z.view(z.size(0), -1)  # [B, N]
            prob = torch.sigmoid(z_flat)
            hard_bits = (prob > 0.5).float()
            z_binarized = hard_bits + prob - prob.detach()  # STE
            # Reshape back to [B, 2c, H, W]
            z_for_channel = z_binarized.view_as(z)
            # Apply BSC
            z_noisy = self._bsc_channel(z_for_channel, self.ber)
            # Map 0/1 → [-1, 1] for decoder (optional but recommended)
            z_input_to_decoder = 2.0 * z_noisy - 1.0
        else:
            # Original analog path
            z_norm = self.encoder.norm(z)  # power normalization
            if self.channel is not None:
                z_input_to_decoder = self.channel(z_norm)
            else:
                z_input_to_decoder = z_norm

        x_hat = self.decoder(z_input_to_decoder)
        return x_hat

    def _bsc_channel(self, x, ber):
        shape = x.shape
        x_flat = x.reshape(shape[0], -1)
        noise = torch.bernoulli(torch.full_like(x_flat, ber))
        x_noisy = (x_flat + noise) % 2
        return x_noisy.reshape(shape)

    def change_channel(self, channel_type='AWGN', snr=None, ber=0.01):
        self.channel_type = channel_type
        if channel_type == 'BSC':
            self.ber = ber
            self.channel = None
        else:
            if snr is not None:
                self.channel = Channel(channel_type, snr)
            else:
                self.channel = None

    def get_channel(self):
        if self.channel_type == 'BSC':
            return self.channel_type, self.ber
        elif hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss


if __name__ == '__main__':
    model = DeepJSCC(c=20)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())
    print(y)
    print(model.encoder.norm)
    print(model.encoder.norm(y))
    print(model.encoder.norm(y).size())
    print(model.encoder.norm(y).size()[1:])
