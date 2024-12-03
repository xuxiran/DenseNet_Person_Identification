import torch


eeglen = 128 # 1 second
decision_window = 128 #1s
device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
batch_size = 128

num_channels = 32

lr=1e-3
weight_decay=0.01

# the length of decision window




