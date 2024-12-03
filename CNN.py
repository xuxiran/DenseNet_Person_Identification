import torch
import torch.nn as nn
import config as cfg



# --------------------------CNN-baseline-----------------------------
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




if __name__ == '__main__':
    model = CNN_baseline()
    print(model)
    model = Model(cfg)
    print(model)
    print("Model is OK")