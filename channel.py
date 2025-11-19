# channel.py
import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20, ber=0.01):
        if channel_type not in ['AWGN', 'Rayleigh', 'BSC']:
            raise Exception('Unknown type of channel: {}'.format(channel_type))
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr
        self.ber = ber

    def forward(self, z_hat):
        if self.channel_type == 'BSC':
            # z_hat should be binary [B, C, H, W] with values in {0,1}
            if z_hat.dim() == 3:
                z_hat = z_hat.unsqueeze(0)  # [1, C, H, W]
            # Flatten to [B, N] for bit-wise flip
            shape = z_hat.shape
            z_flat = z_hat.reshape(shape[0], -1)  # [B, N]

            # BSC: flip each bit with probability ber
            noise = torch.bernoulli(torch.full_like(z_flat, self.ber))
            z_noisy = (z_flat + noise) % 2

            return z_noisy.reshape(shape)
        else:
            # Original AWGN/Rayleigh logic
            if z_hat.dim() not in {3, 4}:
                raise ValueError('Input tensor must be 3D or 4D')
            if z_hat.dim() == 3:
                z_hat = z_hat.unsqueeze(0)

            k = z_hat[0].numel()
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)

            if self.channel_type == 'Rayleigh':
                hc = torch.randn(2, device=z_hat.device)
                z_hat = z_hat.clone()
                half_c = z_hat.size(1) // 2
                z_hat[:, :half_c] = hc[0] * z_hat[:, :half_c]
                z_hat[:, half_c:] = hc[1] * z_hat[:, half_c:]

            return z_hat + noise

    def get_channel(self):
        if self.channel_type == 'BSC':
            return self.channel_type, self.ber
        else:
            return self.channel_type, self.snr