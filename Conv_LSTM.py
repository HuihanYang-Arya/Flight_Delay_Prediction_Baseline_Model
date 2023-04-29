import torch
import torch.nn as nn

class ConvLSTM1DSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, kernel_size=3, padding=1):
        super(ConvLSTM1DSubNet, self).__init__()
        self.conv1d = nn.Conv1d(in_c, hid_c, kernel_size, padding=padding)
        self.lstm = nn.LSTM(hid_c, hid_c, batch_first=True)
        self.fc = nn.Linear(hid_c, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        """
        :param inputs: [B, N, T, C]
        :return: [B, N, T, D]
        """
        B, N, T, C = inputs.size()
        inputs = inputs.view(B * N, T, C)
        inputs = inputs.permute(0, 2, 1)  # Change to [B * N, C, T] for Conv1d
        conv_out = self.conv1d(inputs)
        conv_out = conv_out.permute(0, 2, 1)  # Change back to [B * N, T, C] for LSTM
        lstm_out, _ = self.lstm(conv_out)
        outputs = self.fc(lstm_out)
        outputs = self.act(outputs)
        outputs = outputs.view(B, N, T, -1)
        return outputs

class ConvLSTM1DNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, kernel_size=3, padding=1):
        super(ConvLSTM1DNet, self).__init__()
        self.subnet = ConvLSTM1DSubNet(in_c, hid_c, out_c, kernel_size, padding)

    def forward(self, data, device):
        flow = data["flow_x"]
        flow = flow.to(device)
        prediction = self.subnet(flow)
        return prediction