import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMPeephole(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(LSTMPeephole, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        return x, hidden, cell


class LSTMSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers=2):
        super(LSTMSubNet, self).__init__()
        self.encoder = LSTMPeephole(in_c, hid_c, num_layers=num_layers)
        self.decoder = LSTMPeephole(hid_c, out_c, num_layers=num_layers)
        self.act = nn.ReLU()

    def forward(self, inputs):
        B, N, T, C = inputs.size()
        inputs = inputs.view(B * N, T, C)

        # Encoder
        enc_out,_,_ = self.encoder(inputs)
        # Prepare input for the decoder (using the last encoder output)
        decoder_input = enc_out[:, -1, :].unsqueeze(1).repeat(1, 12, 1)
        # Decoder
        dec_out, _,_ = self.decoder(decoder_input)

        # Activation
        outputs = self.act(dec_out)
        outputs = outputs.view(B, N, -1)
        return outputs

class LSTMNet(nn.Module):
    def __init__(self, in_c, hid_c=256, out_c=1):
        super(LSTMNet, self).__init__()
        self.subnet = LSTMSubNet(in_c, hid_c, out_c)

    def forward(self, data, device):
        flow = data
        flow = flow.to(device)
        prediction = self.subnet(flow)
        return prediction