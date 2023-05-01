import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        #x = x.view(-1, x.size(2), x.size(3))  # Reshape input to (batch_size*num_airports, sequence_length, input_size)
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers)

    def forward(self, x, target_len):
        batch_size = x.size(0)
        num_airports = x.size(1)
        output_size = self.decoder.output_size
        
        # Reshape input to (batch_size*num_airports, sequence_length, input_size)
        x = x.reshape(batch_size * num_airports, -1, 1)
        hidden, cell = self.encoder(x)

        # Prepare input for the decoder
        x = torch.zeros(batch_size * num_airports, 1, output_size).to(x.device)

        # Initialize the output tensor
        outputs = torch.zeros(batch_size * num_airports, target_len, output_size).to(x.device)

        # Decoder loop
        for t in range(target_len):
            x, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = x.squeeze(1)

        # Reshape the output tensor to (batch_size, num_airports, target_len, output_size)
        outputs = outputs.view(batch_size, num_airports, target_len, output_size)

        return outputs
