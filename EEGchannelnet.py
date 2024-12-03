# This is the model presented in the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by 
# Multimodal Learning of Neural Activity and Visual Features,  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)# - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad, 0))

        feature_height = input_height // stride[0]

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1
                )
            ) for i in range(num_residual_blocks)
        ])

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        #print(f"[FE INPUT] {x.shape}")
        out = self.temporal_block(x)
        #print(f"[TEMPORAL] {out.shape}")
        out = self.spatial_block(out)
        #print(f"[SPATIAL] {out.shape}")
        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)
                #print(f"[RESIDUAL] {out.shape}")
        out = self.final_conv(out)
        #print(f"[FINAL] {out.shape}")

        return out

class EEGchannelnet(nn.Module):

    def __init__(self, args):
        super(EEGchannelnet, self).__init__()
        args_defaults=dict(
            in_channels=1,
            temp_channels=10,
            out_channels=50,
            num_classes=26,
            embedding_size=1000,
            input_width=1280,
            input_height=32,
            temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
            temporal_kernel=(1,33),
            temporal_stride=(1,2),
            num_temp_layers=4,
            num_spatial_layers=4,
            spatial_stride=(2,1),
            num_residual_blocks=2,
            down_kernel=3,
            down_stride=2
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.encoder = FeaturesExtractor(
            self.in_channels, self.temp_channels, self.out_channels,
            self.input_width, self.input_height, self.temporal_kernel,
            self.temporal_stride, self.temporal_dilation_list,
            self.num_temp_layers, self.num_spatial_layers,
            self.spatial_stride, self.num_residual_blocks,
            self.down_kernel, self.down_stride)

        encoding_size = self.encoder(torch.zeros(1, self.in_channels, self.input_height, self.input_width)).contiguous().view(-1).size()[0]
        encoding_size = 650
        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, self.embedding_size),
            nn.ReLU(True),
            nn.Linear(self.embedding_size, self.num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        #print(f"[INPUT] {x.shape}")
        out = self.encoder(x)
        #print(f"[ENCODER] {out.shape}")
        out = out.view(x.size(0), -1)
        #print(f"[FLATTEN] {out.shape}")
        out = self.classifier(out)
        #print(f"[CLASSIFIER] {out.shape}")
        return out

if __name__ == '__main__':
    model = EEGchannelnet({})
    print(model)
    x = torch.randn(128, 128, 32)
    print(model(x).shape)
    print("Model is working!")