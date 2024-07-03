import torch
import torch.nn as nn

class BaselineSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size):
        super(BaselineSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, src, tgt):
        embedded = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded)
        
        decoder_input = self.embedding(tgt[:, :-1])
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoder_output)
        return output

    def encode(self, src):
        embedded = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded)
        return hidden, cell

    def decode_step(self, input, hidden, cell):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell