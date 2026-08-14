import torch.nn as nn
from torchinfo import summary

def conv_block(in_channels, out_channels, kernel_size=3, num_convs=2):
    layers = []
    for i in range(num_convs):
        layers.append(
            nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
        )
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SegNet3D(nn.Module):
    def __init__(self, in_channels, n_labels=1, kernel=3, pool_size=2):
        super(SegNet3D, self).__init__()

        # -------- Encoder --------
        self.enc1 = conv_block(in_channels, 32, kernel, num_convs=2)
        self.enc2 = conv_block(32, 64, kernel, num_convs=2)
        self.enc3 = conv_block(64, 128, kernel, num_convs=3)
        self.enc4 = conv_block(128, 256, kernel, num_convs=3)
        self.enc5 = conv_block(256, 256, kernel, num_convs=3)

        self.pool = nn.MaxPool3d(pool_size, stride=pool_size, return_indices=True)
        self.unpool = nn.MaxUnpool3d(pool_size, stride=pool_size)

        # -------- Decoder --------
        self.dec5 = conv_block(256, 256, kernel, num_convs=3)
        self.dec4 = conv_block(256, 128, kernel, num_convs=3)
        self.dec3 = conv_block(128, 64, kernel, num_convs=3)
        self.dec2 = conv_block(64, 32, kernel, num_convs=2)
        self.dec1 = conv_block(32, 32, kernel, num_convs=1)

        self.final_conv = nn.Conv3d(32, n_labels, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):

        # -------- Encoder --------
        x1 = self.enc1(x)
        size1 = x1.size()
        x1p, idx1 = self.pool(x1)

        x2 = self.enc2(x1p)
        size2 = x2.size()
        x2p, idx2 = self.pool(x2)

        x3 = self.enc3(x2p)
        size3 = x3.size()
        x3p, idx3 = self.pool(x3)

        x4 = self.enc4(x3p)
        size4 = x4.size()
        x4p, idx4 = self.pool(x4)

        x5 = self.enc5(x4p)
        size5 = x5.size()
        x5p, idx5 = self.pool(x5)

        # -------- Decoder --------
        d5 = self.unpool(x5p, idx5, output_size=size5)
        d5 = self.dec5(d5)

        d4 = self.unpool(d5, idx4, output_size=size4)
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, idx3, output_size=size3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=size2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=size1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)