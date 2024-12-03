import torch
import torch.nn as nn
import config as cfg

"""
@INPROCEEDINGS{10448013,
  author={Xu, Xiran and Wang, Bo and Yan, Yujie and Wu, Xihong and Chen, Jing},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A DenseNet-Based Method for Decoding Auditory Spatial Attention with EEG}, 
  year={2024},
  volume={},
  number={},
  pages={1946-1950},
  keywords={Three-dimensional displays;Two-dimensional displays;Feature extraction;Electroencephalography;Decoding;Recording;Reverberation;Auditory attention decoding;auditory spatial attention detection;EEG;brain lateralization;DenseNet},
  doi={10.1109/ICASSP48485.2024.10448013}}
  """

channel_name = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1',
                'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
# --------------------------DenseNet-----------------------------
# --------------------------Map-----------------------------
#    0   1   2   3   4   5   6   7   8   9   10
# 0  N   N   N   N   Fp1 N   Fp2 N   N   N   N
# 1  N   F7  N   F3  N   Fz  N   F4  N   F8  N
# 2  FT9 N   FC5 N   FC1 N   FC2 N   FC6 N   FT10
# 3  N   T7  N   C3  N   Cz  N   C4  N   N   T8
# 4  N   N   CP5 N   CP1 N   CP2 N   CP6 N   N
# 5  N   P7  N   P3  N   Pz  N   P4  N   P8  N
# 6  N   PO9 N   O1  N   Oz  N   O2  N  PO10 N
Map = [  ['N', 'N', 'N', 'N', 'Fp1', 'N', 'Fp2', 'N', 'N', 'N', 'N'],
         ['N', 'F7', 'N', 'F3', 'N', 'Fz', 'N', 'F4', 'N', 'F8', 'N'],
         ['FT9', 'N', 'FC5', 'N', 'FC1', 'N', 'FC2', 'N', 'FC6', 'N', 'FT10'],
         ['N', 'T7', 'N', 'C3', 'N', 'Cz', 'N', 'C4', 'N', 'N', 'T8'],
         ['N', 'N', 'CP5', 'N', 'CP1', 'N', 'CP2', 'N', 'CP6', 'N', 'N'],
         ['N', 'P7', 'N', 'P3', 'N', 'Pz', 'N', 'P4', 'N', 'P8', 'N'],
         ['N', 'PO9', 'N', 'O1', 'N', 'Oz', 'N', 'O2', 'N', 'PO10', 'N']]

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math

class DenseBlock(nn.Module):
    def conv_block(self,in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm3d(in_channels),
                            nn.ReLU(),
                            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1)))
        return blk

    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels

            # padding
            # pad_time = nn.ReplicationPad3d((0, 0, 0, 1, 1))
            # net.append(pad_time)
            net.append(self.conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # get the out channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # concat the input dim and output dim
        return X


class DenseNet(nn.Module):
    def __init__(self,channel_num=16):
        super(DenseNet, self).__init__()
        self.num_channels = 64
        self.growth_rate = 32
        self.feature = self.densenet(self.num_channels)
        self.linear = nn.Linear(248, 26)


    def transition_block(self, in_channels, out_channels):
        blk = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            nn.AvgPool3d(kernel_size=(7, 2, 2), stride=(3, 1, 1))
        )
        return blk

    def densenet(self,channel_num=16):
        net = nn.Sequential(
            nn.Conv3d(1, self.num_channels, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(self.num_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        )

        num_channels, growth_rate = self.num_channels, self.growth_rate  # num_channels is the currenct channels
        num_convs_in_dense_blocks = [4,4,4,4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            net.add_module("DenseBlosk_%d" % i, DB)
            # last channel
            num_channels = DB.out_channels
            # reduce the output channel
            if i != len(num_convs_in_dense_blocks) - 1:
                net.add_module("transition_block_%d" % i, self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        net.add_module("BN", nn.BatchNorm3d(num_channels))
        net.add_module("relu", nn.ReLU())
        return net


    def forward(self, x):
        # Layer 1
        x_size = x.size()
        x_new = torch.zeros(x_size[0], x_size[1], 7, 11).to(cfg.device)

        for i in range(7):
            for j in range(11):
                if Map[i][j] != 'N':
                    x_new[:, :, i, j] = x[:, :, channel_name.index(Map[i][j])]
        x = x_new
        x = x.unsqueeze(dim=1)

        x = self.feature(x)
        x = F.avg_pool3d(x, kernel_size=x.size()[2:])
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x



if __name__ == '__main__':

    x = torch.rand(128, 128, 32)

    model = DenseNet()
    y = model(x)
    print(y.shape)

