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
        return torch.zeros(batch_size, self.hidden_size).cuda(device)


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
        return torch.zeros(1, batch_size, self.hidden_size).cuda(device)


class UtteranceEncoder(nn.Module):
    def __init__(self, utt_input_size, embed_size, utterance_hidden, padding_idx, fine_tuning=False):
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
        X = torch.tanh(self.eh(self.xe(X))) # (batch_size, 1, seq_len, embed_size)
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
        return torch.zeros(2, batch_size, self.hidden_size).cuda(device)


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
        return torch.zeros(1, batch_size, self.hidden_size).cuda(device)


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
        self.th = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, Y, hidden, tag=None):
        h = torch.tanh(self.eh(self.ye(Y)))
        if not tag is None:
            h = self.th(torch.cat((h, tag), dim=2))
        output, hidden = self.hh(h, hidden)
        y_dist = self.ey(torch.tanh(self.he(output.squeeze(1))))
        return y_dist, hidden, output


class DAPairEncoder(nn.Module):
    def __init__(self, da_hidden_size, da_embed_size, da_vocab_size):
        super(DAPairEncoder, self).__init__()
        self.hidden_size = da_hidden_size
        self.embed_size = da_embed_size
        self.vocab_size = da_vocab_size

        self.te = nn.Embedding(self.vocab_size, self.embed_size)
        self.eh = nn.Linear(self.embed_size * 2, self.hidden_size)

    def forward(self, X):
        embeded = self.te(X)
        c, n, _, _ = embeded.size()
        embeded = embeded.view(c, n, -1)
        return torch.tanh(self.eh(embeded))

class OrderReasoningLayer(nn.Module):
    def __init__(self, encoder_hidden_size, hidden_size, middle_layer_size, da_hidden_size, config):
        super(OrderReasoningLayer, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.middle_layer_size = middle_layer_size
        self.da_hidden_size = da_hidden_size
        self.config = config

        self.max_pooling = ChannelPool(kernel_size=3)
        self.xh = nn.Linear(self.encoder_hidden_size * 2, self.hidden_size)
        self.hh = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=False)
        self.hh_b = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=False)
        self.tt = nn.GRU(self.da_hidden_size, self.da_hidden_size)
        self.hm = nn.Linear(self.hidden_size * 6, self.middle_layer_size)
        if self.config['use_da']:
            self.mm = nn.Linear(self.middle_layer_size + self.da_hidden_size * 3, self.middle_layer_size)
        self.my = nn.Linear(self.middle_layer_size, 2)

    def forward(self, XOrdered, XMisOrdered, XTarget,
                DAOrdered, DAMisOrdered, DATarget, Y, hidden, da_hidden):
        XOrdered = self.xh(XOrdered)
        XMisOrdered = self.xh(XMisOrdered)
        XTarget = self.xh(XTarget)
        O_output, _ = self.hh(XOrdered, hidden)
        O_output_b, _ = self.hh_b(self._invert_tensor(XOrdered), hidden)
        M_output, _ = self.hh(XMisOrdered, hidden)
        M_output_b, _ = self.hh_b(self._invert_tensor(XMisOrdered), hidden)
        T_output, _ = self.hh(XTarget, hidden)
        T_output_b, _ = self.hh_b(self._invert_tensor(XTarget), hidden)
        # output: (window_size, batch_size, hidden_size)

        if self.config['use_da']:
            da_o_output, _ = self.tt(DAOrdered, da_hidden)
            da_m_output, _ = self.tt(DAMisOrdered, da_hidden)
            da_t_output, _ = self.tt(DATarget, da_hidden)
            da_output = torch.cat((da_o_output[-1], da_m_output[-1], da_t_output[-1]), dim=-1)

        O_output = torch.cat((self.max_pooling.forward(O_output), self.max_pooling.forward(O_output_b)), dim=1)
        M_output = torch.cat((self.max_pooling.forward(M_output), self.max_pooling.forward(M_output_b)), dim=1)
        T_output = torch.cat((self.max_pooling.forward(T_output), self.max_pooling.forward(T_output_b)), dim=1)
        # output: (batch_size, hidden_size * 2)
        output = torch.cat((O_output, M_output, T_output), dim=-1)
        # output: (batch_size, hidden_size * 6)
        if self.config['use_da']:
            output = self.mm(torch.cat((self.hm(output), da_output), dim=-1))
        else:
            output = self.hm(output)
        pred = self.my(output)
        return pred

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size).cuda(device)

    def initDAHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.da_hidden_size).cuda(device)

    def _invert_tensor(self, X):
        return X[torch.arange(X.size(0)-1, -1, -1)]


class ChannelPool(nn.MaxPool1d):
    def forward(self, X):
        X = X.permute(1,2,0)
        pooled = F.max_pool1d(X, self.kernel_size)
        pooled = pooled.permute(2,0,1).squeeze(0)
        return pooled


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
