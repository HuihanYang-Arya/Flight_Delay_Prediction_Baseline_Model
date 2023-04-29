import torch
import torch.nn as nn

class GRUSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GRUSubNet, self).__init__()
        self.gru = nn.GRU(in_c, hid_c, batch_first=True)
        self.fc = nn.Linear(hid_c, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        """
        :param inputs: [B, N, T, C]
        :return: [B, N, T, D]
        """
        B, N, T, C = inputs.size()
        inputs = inputs.view(B * N, T, C)
        gru_out, _ = self.gru(inputs)
        outputs = self.fc(gru_out)
        outputs = self.act(outputs)
        outputs = outputs.view(B, N, T, -1)
        return outputs

class GRUNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GRUNet, self).__init__()
        self.subnet = GRUSubNet(in_c, hid_c, out_c)

    def forward(self, data, device):
        flow = data["flow_x"]
        flow = flow.to(device)
        prediction = self.subnet(flow)
        return prediction