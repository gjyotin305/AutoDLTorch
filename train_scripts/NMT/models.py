# Import Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import MAX_LENGTH, SOS_TOKEN
from einops import rearrange

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.gru(embedded)
        return out, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, out_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.embedding = nn.Embedding(self.out_size, hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.out_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.out_size)
    
    def forward_step(self, x, x_h):
        out = self.embedding(x)
        out = F.relu(out)
        out, hid = self.gru(out, x_h)
        out = F.softmax(out)
        return out, hid
    
    def forward(self, enc_out, enc_hid, target_tensor=None):
        batch_size = enc_out.size(0)
        decoder_in= torch.empty(
            batch_size, 
            1, 
            dtype=torch.long
        ).fill_(SOS_TOKEN)
        decoder_hid = enc_hid
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_out, decoder_hid = self.forward_step(
                decoder_in, 
                decoder_hid
            )
            decoder_outputs.append(decoder_out)
            
            if target_tensor:
                decoder_in = target_tensor[:, i].unsqueeze(1) # Teacher Forcing
            else:
                _, topi = decoder_out.topk(1)
                decoder_in = topi.squeeze(-1).detach()
            
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            return decoder_outputs, decoder_hid, None
