"""
Bi-LSTM classifier model

@author: Zhenwei Wang
@date: 12/30/2019
"""

import torch.nn as nn
import torch


class RNN(nn.Module):
    """
    Our three layers are:
        - embedding layer: sparse one-hot vector -> dense embedding vector
        - Bi-LSTM: dense vector + hidden state h_t-1 -> hidden state h_t
        - linear layer: final hidden state -> output dimension (class num)
    """

    def __init__(self, input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()
        # input_dim: vocabulary size
        # embedding_dim: depend on pretrained word2vec
        # hidden_dim: usually equal to emd_dim
        # output_dim: number of classes
        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim,
                                      padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sentence length, batch size]
        # You may notice that this tensor should have another dimension due to the one-hot vectors
        # however PyTorch conveniently stores a one-hot vector as it's index value

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)