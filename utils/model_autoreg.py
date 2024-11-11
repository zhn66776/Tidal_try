import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 定义模型
# 简单BiLSTM
class BiLSTM_Reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.5):
        super(BiLSTM_Reg, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)  # bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch_size, seq_len, feature_size)

        out = self.fc(out[:, -1, :])  # 使用隐藏层最后一个time step进行预测

        return out



##11用预测值去预测 凌晨
class CNNBiLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(CNNBiLSTM, self).__init__()

        # 您可以根据需要调整 ConvModule 的层数和参数
        self.conv_pool = nn.Sequential(ConvModule())
        self.lstm = BiLSTMModule(hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x):
        x = self.conv_pool(x)  # x 形状: (batch_size, channel_size=1, seq_len)
        
        # 确保只移除 channel_size 为 1 的维度，防止批量大小为 1 时出错
        x = x.squeeze(dim=1)  # x 形状: (batch_size, seq_len)
        x = x.unsqueeze(-1)   # x 形状: (batch_size, seq_len, feature_size=1)
        
        out = self.lstm(x)    # 输出形状取决于 LSTM 的设置
        return out


class BiLSTMModule(nn.Module):
    def __init__(self, hidden_size=1, num_layers=2, dropout=0.5):
        super(BiLSTMModule, self).__init__()
        # 在这里你可以修改成GRU单元
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, 1)  # bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch_size, seq_len, feature_size)

        out = self.fc(out[:, -1, :])  # 使用隐藏层最后一个time step进行预测

        return out


# TCN模块
# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# TCN主网络
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
