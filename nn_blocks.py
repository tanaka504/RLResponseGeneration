import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DAEncoder(nn.Module):
    def __init__(self, da_input_size, da_embed_size,da_hidden):
        super(DAEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.xe = nn.Embedding(da_input_size, da_embed_size)
        self.eh = nn.Linear(da_embed_size, da_hidden)

    def forward(self, DA):
        embedding = torch.tanh(self.eh(self.xe(DA))) # (batch_size, 1) -> (batch_size, 1, hidden_size)
        return embedding

    def initHidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device)


class DAContextEncoder(nn.Module):
    def __init__(self, da_hidden):
        super(DAContextEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.hh = nn.GRU(da_hidden, da_hidden, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, batch_size, device):
        # h_0 = (num_layers * num_directions, batch_size, hidden_size)
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class UtteranceEncoder(nn.Module):
    def __init__(self, utt_input_size, embed_size, utterance_hidden, padding_idx, fine_tuning):
        super(UtteranceEncoder, self).__init__()
        self.hidden_size = utterance_hidden
        self.padding_idx = padding_idx
        self.xe = nn.Embedding(utt_input_size, embed_size)
        self.xe.weight.requires_grad = False if fine_tuning else True
        self.eh = nn.Linear(embed_size, utterance_hidden)
        self.hh = nn.GRU(utterance_hidden, utterance_hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, X, hidden):
        lengths = (X != self.padding_idx).sum(dim=1)
        seq_len, sort_idx = lengths.sort(descending=True)
        _, unsort_idx = sort_idx.sort(descending=False)
        # sorting
        X = F.tanh(self.eh(self.xe(X))) # (batch_size, 1, seq_len, embed_size)
        sorted_X = X[sort_idx]
        # padding
        packed_tensor = pack_padded_sequence(sorted_X, seq_len, batch_first=True)
        output, hidden = self.hh(packed_tensor, hidden)
        # unpacking
        output, _ = pad_packed_sequence(output, batch_first=True)
        # extract last timestep output
        idx = (lengths - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
        output = output.gather(1, idx)
        # unsorting
        output = output[unsort_idx]
        hidden = hidden[:, unsort_idx]

        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(2, batch_size, self.hidden_size).to(device)


class UtteranceContextEncoder(nn.Module):
    def __init__(self, utterance_hidden_size):
        super(UtteranceContextEncoder, self).__init__()
        self.hidden_size = utterance_hidden_size
        self.hh = nn.GRU(utterance_hidden_size, utterance_hidden_size, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

class UtteranceDecoder(nn.Module):
    def __init__(self, utterance_hidden_size, utt_embed_size, utt_vocab_size):
        super(UtteranceDecoder, self).__init__()
        self.hidden_size = utterance_hidden_size
        self.embed_size = utt_embed_size
        self.vocab_size = utt_vocab_size

        self.ye = nn.Embedding(self.vocab_size, self.embed_size)
        self.eh = nn.Linear(self.embed_size, self.hidden_size)
        self.hh = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.he = nn.Linear(self.hidden_size, self.embed_size)
        self.ey = nn.Linear(self.embed_size, self.vocab_size)
        self.th = nn.Linear(self.hidden_size + 10, self.hidden_size)

    def forward(self, Y, hidden, tag=None):
        h = F.tanh(self.eh(self.ye(Y)))
        if not tag is None:
            h = self.th(torch.cat((h, tag), dim=2))
        output, hidden = self.hh(h, hidden)
        y_dist = self.ey(torch.tanh(self.he(output.squeeze(1))))
        return y_dist, hidden, output

class BeamNode(object):
    def __init__(self, hidden, previousNode, wordId, logProb, length):
        self.hidden = hidden
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.eval() < other.eval()
