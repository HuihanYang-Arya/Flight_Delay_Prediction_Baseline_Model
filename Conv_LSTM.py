import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.convlstm = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)

    def forward(self, x):
        hidden = self.convlstm(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, kernel_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.convlstm = nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=1)

    def forward(self, x, hidden):
        output = self.convlstm(hidden)
        return output

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, kernel_size, num_layers=2):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, kernel_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, kernel_size, num_layers)

    def forward(self, x, target_len):
        batch_size = x.size(0)
        num_airports = x.size(1)
        output_size = 1
        
        # Reshape input to (batch_size*num_airports, input_size, sequence_length)
        x = x.reshape(batch_size * num_airports, 1, -1)
        hidden = self.encoder(x)

        # Decoder
        outputs = self.decoder(x, hidden)
        print(outputs.shape)
        # Reshape the output tensor to (batch_size, num_airports, output_size, target_len)
        outputs = outputs.view(batch_size, num_airports, 1, target_len)

        return outputs
