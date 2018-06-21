import numpy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embed_regularize import embedded_dropout
from weight_drop import WeightDrop


class Shuffle(nn.Module):
    def __init__(self, permutation, contiguous=True):
        super(Shuffle, self).__init__()
        self.permutation = permutation
        self.contiguous = contiguous

    def forward(self, input):
        shuffled = input.permute(*self.permutation)
        if self.contiguous:
            return shuffled.contiguous()
        else:
            return shuffled


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class distance_parser(nn.Module):
    def __init__(self,
                 vocab_size, embed_size, hid_size,
                 arc_size, stag_size, window_size,
                 wordembed=None, dropout=0.2, dropoute=0.1, dropoutr=0.1):
        super(distance_parser, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hid_size = hid_size
        self.arc_size = arc_size
        self.stag_size = stag_size
        self.window_size = window_size
        self.drop = nn.Dropout(dropout)
        self.dropoute = dropoute
        self.dropoutr = dropoutr
        self.encoder = nn.Embedding(vocab_size, embed_size)
        if wordembed is not None:
            self.encoder.weight.data = torch.FloatTensor(wordembed)

        self.tag_encoder = nn.Embedding(stag_size, embed_size)

        self.word_rnn = nn.LSTM(2 * embed_size, hid_size, num_layers=2, batch_first=True, dropout=dropout,
                                bidirectional=True)
        self.word_rnn = WeightDrop(self.word_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.conv1 = nn.Sequential(nn.Dropout(dropout),
                                   nn.Conv1d(hid_size * 2,
                                             hid_size,
                                             window_size),
                                   nn.ReLU())

        self.arc_rnn = nn.LSTM(hid_size, hid_size, num_layers=2, batch_first=True, dropout=dropout,
                               bidirectional=True)
        self.arc_rnn = WeightDrop(self.arc_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.distance = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_size * 2, hid_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, 1),
        )

        self.terminal = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_size * 2, hid_size),
            nn.ReLU(),
        )

        self.non_terminal = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_size * 2, hid_size),
            nn.ReLU(),
        )

        self.arc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_size, arc_size),
        )

    def forward(self, words, stag, mask):
        """
        tokens: Variable of LongTensor, shape (bsize, ntoken,)
        mock_emb: mock embedding for convolution overhead
        """

        bsz, ntoken = words.size()
        emb_words = embedded_dropout(self.encoder, words, dropout=self.dropoute if self.training else 0)
        emb_words = self.drop(emb_words)

        emb_stags = embedded_dropout(self.tag_encoder, stag, dropout=self.dropoute if self.training else 0)
        emb_stags = self.drop(emb_stags)


        def run_rnn(input, rnn, lengths):
            sorted_idx = numpy.argsort(lengths)[::-1].tolist()
            rnn_input = pack_padded_sequence(input[sorted_idx], lengths[sorted_idx], batch_first=True)
            rnn_out, _ = rnn(rnn_input)  # (bsize, ntoken, hidsize*2)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
            rnn_out = rnn_out[numpy.argsort(sorted_idx).tolist()]

            return rnn_out

        sent_lengths = (mask.sum(dim=1)).data.cpu().numpy().astype('int')
        dst_lengths = sent_lengths - 1
        emb_plus_tag = torch.cat([emb_words, emb_stags], dim=-1)

        rnn1_out = run_rnn(emb_plus_tag, self.word_rnn, sent_lengths)

        terminal = self.terminal(rnn1_out.view(-1, self.hid_size*2))
        tag = self.arc(terminal)  # (bsize, ndst, tagsize)

        conv_out = self.conv1(rnn1_out.permute(0, 2, 1)).permute(0, 2, 1)  # (bsize, ndst, hidsize)
        rnn2_out = run_rnn(conv_out, self.arc_rnn, dst_lengths)

        non_terminal = self.non_terminal(rnn2_out.view(-1, self.hid_size*2))
        distance = self.distance(rnn2_out.view(-1, self.hid_size*2)).squeeze(dim=-1)  # (bsize, ndst)
        arc = self.arc(non_terminal)  # (bsize, ndst, arcsize)
        return distance.view(bsz, ntoken - 1), arc.contiguous().view(-1, self.arc_size), tag.view(-1, self.arc_size)
