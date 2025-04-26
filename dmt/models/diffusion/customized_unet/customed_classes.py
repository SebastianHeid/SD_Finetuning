import torch.nn as nn


# -------------- Upsample Blocks --------------
class IdentityBlock0Up(nn.Module):
    def __init__(self, in_channels=2560, out_channels=1280):
        super(IdentityBlock0Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp1Res2(nn.Module):
    def __init__(self, in_channels=1920, out_channels=1280):
        super(IdentityBlockUp1Res2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp2Res0(nn.Module):
    def __init__(self, in_channels=1920, out_channels=640):
        super(IdentityBlockUp2Res0, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp2Res1(nn.Module):
    def __init__(self, in_channels=1280, out_channels=640):
        super(IdentityBlockUp2Res1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp2Res2(nn.Module):
    def __init__(self, in_channels=960, out_channels=640):
        super(IdentityBlockUp2Res2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp3Res0(nn.Module):
    def __init__(self, in_channels=960, out_channels=320):
        super(IdentityBlockUp3Res0, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class IdentityBlockUp3Res(nn.Module):
    def __init__(self, in_channels=640, out_channels=320):
        super(IdentityBlockUp3Res, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


# -------------- Downsample Blocks --------------


class IdentityBlockAttn(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x.unsqueeze(0)


class IdentityBlock(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class Down2Res1(nn.Module):
    def __init__(self, in_channels=320, out_channels=640):
        super(Down2Res1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class Down3Res1(nn.Module):
    def __init__(self, in_channels=640, out_channels=1280):
        super(Down3Res1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)
