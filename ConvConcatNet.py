import torch
import torch.nn as nn
import config as cfg
"""
@INPROCEEDINGS{10626859,
  author={Xu, Xiran and Wang, Bo and Yan, Yujie and Zhu, Haolin and Zhang, Zechen and Wu, Xihong and Chen, Jing},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={ConvConcatNet: A Deep Convolutional Neural Network to Reconstruct Mel Spectrogram from the EEG}, 
  year={2024},
  volume={},
  number={},
  pages={113-114},
  keywords={Training;Correlation;Brain modeling;Electroencephalography;Convolutional neural networks;Task analysis;Speech processing;Mel spectrogram reconstruction;EEG;ConvConcatNet;unseen subject;unseen stimuli},
  doi={10.1109/ICASSPW62465.2024.10626859}}
"""
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()
        self.layernorm = nn.LayerNorm(cfg.num_channels)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(17,cfg.num_channels), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))

        # We are sorry the code is not nice here
        # But the para is very important

        self.fc1 = nn.Linear(in_features=100, out_features=100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=100, out_features=26)

    def forward(self, x):
        x = self.layernorm(x)
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)

        return fc2_out

class Satt(nn.Module):
    def __init__(self):
        super(Satt, self).__init__()
        self.linear1 = nn.Linear(cfg.num_channels, cfg.num_channels)
        self.linear2 = nn.Linear(cfg.num_channels, cfg.num_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, E):

        M_s = self.linear2(self.relu(self.linear1(E)))


        return M_s

class Extractor(nn.Module):
    def __init__(self, input_channels=128):
        super(Extractor, self).__init__()
        self.convs1 = nn.Conv1d(cfg.num_channels*3, cfg.num_channels, 1)
        self.convt1 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels*4, 9,groups=cfg.num_channels*4)

        self.convs2 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels, 1)
        self.convt2 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels*4, 9,groups=cfg.num_channels*4)

        self.convs3 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels, 1)
        self.convt3 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels*4, 9,groups=cfg.num_channels*4)

        self.convs4 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels, 1)
        self.convt4 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels*4, 9,groups=cfg.num_channels*4)

        self.conv5 = nn.Conv1d(cfg.num_channels*4, cfg.num_channels*2, 9)

        self.norm1 = nn.LayerNorm(cfg.num_channels)
        self.norm2 = nn.LayerNorm(cfg.num_channels*2)
        self.norm3 = nn.LayerNorm(cfg.num_channels*4)

        self.relu = nn.LeakyReLU()
        self.pad = nn.ZeroPad2d((0, 0, 4, 4))

    def forward(self, x):
        eeg = x

        x = x.permute(0, 2, 1)
        x = self.convs1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt1(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs2(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt2(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs3(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs4(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt4(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pad(x)
        return x

class OutputContext(nn.Module):
    def __init__(self, input_channels=cfg.num_channels):
        super(OutputContext, self).__init__()

        self.pad = nn.ZeroPad2d((0, 0, 19, 19))
        self.conv = nn.Conv1d(input_channels, cfg.num_channels, 39)
        self.norm = nn.LayerNorm(cfg.num_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pad(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ConvConcatNet(nn.Module):
    def __init__(self, nb_blocks=6, input_channels=cfg.num_channels, output_dim=1):
        super(ConvConcatNet, self).__init__()

        self.extractor = Extractor(input_channels=input_channels)
        self.satt = Satt()
        self.output_context = OutputContext()
        self.nb_blocks = nb_blocks
        self.linear_layer = nn.Linear(cfg.num_channels*2, cfg.num_channels)
        self.fc = CNN_baseline()

    def forward(self, x):
        # copy the 2 dim
        eeg = x
        eeg_out = torch.zeros_like(eeg)
        eeg_out_att = torch.zeros_like(eeg)
        for i in range(self.nb_blocks):

            eeg_out = torch.cat((eeg,eeg_out,eeg_out_att), dim=2)

            eeg_out = self.extractor(eeg_out)

            eeg_out = self.linear_layer(eeg_out)
            eeg_out = self.output_context(eeg_out)

            eeg_out_att = self.satt(eeg_out)
            eeg_out_att = eeg_out_att * eeg_out

        eeg_out = self.fc(eeg_out.squeeze(dim=2))

        return eeg_out

if __name__ == '__main__':
    model = ConvConcatNet()
    x = torch.rand(128, 128, 34)
    y = model(x)
    print(model(x).shape)
