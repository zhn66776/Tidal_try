import torch.nn as nn
class ConvModule(nn.Module):
    """
    convolution-based options:

    conv-1d:
    (batch_size, channel_size, input_size)  input_size(seq_len)是conv-1d的操作维度
    after conv:
    seq_len' = (input_size - kernel_size + 2 * pad_size) // stride + 1
    (k=3, s=1, p=1) 可以保持输入维度不变
    """

    def __init__(self,
                 c_k=3, c_s=1, c_p=1,  # conv param
                 p_k=3, p_s=1, p_p=1,  # pool param
                 if_pool: bool = True):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=c_k,
                              stride=c_s,
                              padding=c_p)

        self.if_pool = if_pool
        if if_pool:
            self.pool = nn.MaxPool1d(kernel_size=p_k,
                                     stride=p_s,
                                     padding=p_p)

    def forward(self, x):
        out = self.conv(x)

        if self.if_pool:
            out = self.pool(out)

        return out
class CNNBiLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(CNNBiLSTM, self).__init__()

        self.conv_pool = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1),  # 2输入特征
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = BiLSTMModule(hidden_size=hidden_size, num_layers=num_layers, input_size=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度适配Conv1d
        x = self.conv_pool(x)
        x = x.permute(0, 2, 1)  # 调整维度适配LSTM
        out = self.lstm(x)
        return out


class BiLSTMModule(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size=2, dropout=0.5):
        super(BiLSTMModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向LSTM

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个time step的输出
        return out
