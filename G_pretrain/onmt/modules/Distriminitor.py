import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt
import random
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.Utils import formalize, deformalize
from onmt.modules.Conv2Conv import CNNEncoder, StackedCNN, shape_transform

SCALE_WEIGHT = 0.5 ** 0.5


class Disc(nn.Module):

    def __init__(self, d_input_size, hidden_size, embeddings, _):
        super(Disc, self).__init__()
        self.embeddings = embeddings
        self.src_rnn = nn.GRU(d_input_size, hidden_size=hidden_size, dropout=0.2)
        self.tgt_rnn = nn.GRU(d_input_size, hidden_size=hidden_size, dropout=0.2)
        self.ref_rnn = nn.GRU(d_input_size, hidden_size=2, dropout=0.2)
        self.project1 = nn.Linear(hidden_size, 64)
        self.project2 = nn.Linear(64, 2)
        self.dp = nn.Dropout(p=0.2)
        self.sf = nn.Softmax(dim=-1)

    def encode(self, src, src_lengths, ref):
        src_emb = self.embeddings(src)
        packed_src_emb = pack(src_emb, src_lengths.tolist())
        src_output, src_hidden = self.src_rnn(packed_src_emb)

        ref_hidden = []
        ref_context = []
        ref_lengths = []

        for r, r_len in ref:
            r, r_len_sorted, origin_new = formalize(r, r_len)

            r_emb = self.embeddings(r.unsqueeze(2))
            packed_r_emb = pack(r_emb, r_len_sorted.view(-1).tolist())

            r_output, r_hidden = self.ref_rnn(packed_r_emb)
            r_output, _ = unpack(r_output)

            r_output = deformalize(r_output, origin_new)
            r_hidden = deformalize(r_hidden, origin_new)

            ref_hidden.append(r_hidden)
            ref_context.append(r_output)
            ref_lengths.append(r_len)

        return src_hidden, ref_hidden, ref_context, ref_lengths

    def forward(self, tgt, tgt_lengths, src, src_lengths, ref, step_type='d_step'):

        src_hidden, ref_hidden, ref_context, ref_lengths = self.encode(src, src_lengths, ref)
        # tgt_lengths_list = tgt_lengths.data.tolist()
        tgt, tgt_lengths_sorted, origin_new = formalize(tgt, tgt_lengths)

        tgt_emb = self.embeddings(tgt)
        packed_tgt_emb = pack(tgt_emb, tgt_lengths_sorted.view(-1).tolist())

        tgt_output, tgt_hidden = self.tgt_rnn(packed_tgt_emb, src_hidden)
        tgt_output, _ = unpack(tgt_output)

        tgt_output = deformalize(tgt_output, origin_new)
        tgt_hidden = deformalize(tgt_hidden, origin_new)
        # logits = self.project1(torch.cat([ref_attn_tgt.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
        logits = self.project1(torch.cat([tgt_hidden.squeeze(0)], dim=-1))
        logits = self.project2(self.dp(F.tanh(logits)))

        if step_type == 'self_sample':
            logits = logits.unsqueeze(0).repeat(tgt_output.size(0), 1, 1)

        return logits


class NLI(nn.Module):

    def __init__(self, d_input_size, hidden_size, embeddings, _):
        super(NLI, self).__init__()
        self.embeddings = embeddings
        self.src_rnn = nn.GRU(d_input_size, hidden_size=hidden_size, dropout=0.2)
        self.tgt_rnn = nn.GRU(d_input_size, hidden_size=hidden_size, dropout=0.2)
        self.ref_rnn = nn.GRU(d_input_size, hidden_size=hidden_size, dropout=0.2)
        self.project1 = nn.Linear(hidden_size * 2, 64)
        self.project2 = nn.Linear(64, 3)
        self.dp = nn.Dropout(p=0.2)
        self.sf = nn.Softmax(dim=-1)

    def encode(self, src, src_lengths, per):
        src_emb = self.embeddings(src)
        packed_src_emb = pack(src_emb, src_lengths.tolist())
        src_output, src_hidden = self.src_rnn(packed_src_emb)

        per_hidden = []
        per_context = []
        per_lengths = []

        for p, p_len in per:
            p, p_len_sorted, origin_new = formalize(p, p_len)
            p_emb = self.embeddings(p.unsqueeze(2))
            packed_r_emb = pack(p_emb, p_len_sorted.view(-1).tolist())
            p_output, p_hidden = self.ref_rnn(packed_r_emb)
            p_output, _ = unpack(p_output)
            p_output = deformalize(p_output, origin_new)
            p_hidden = deformalize(p_hidden, origin_new)
            per_hidden.append(p_hidden)
            per_context.append(p_output)
            per_lengths.append(p_len)

        return src_hidden, per_hidden, per_context, per_lengths

    def encode_nli(self, nli_data):
        src, src_len = nli_data[0]
        tgt, tgt_len = nli_data[1]

        src, src_len_sorted, src_origin_new = formalize(src, src_len)
        src_emb = self.embeddings(src.unsqueeze(2))
        packed_src_emb = pack(src_emb, src_len_sorted.view(-1).tolist())
        _, src_hidden = self.ref_rnn(packed_src_emb)
        src_hidden = deformalize(src_hidden, src_origin_new)

        tgt, tgt_len_sorted, tgt_origin_new = formalize(tgt, tgt_len)
        tgt_emb = self.embeddings(tgt.unsqueeze(2))
        packed_tgt_emb = pack(tgt_emb, tgt_len_sorted.view(-1).tolist())
        _, tgt_hidden = self.ref_rnn(packed_tgt_emb)
        tgt_hidden = deformalize(tgt_hidden, tgt_origin_new)

        return src_hidden, tgt_hidden


    def forward(self, tgt, tgt_lengths, src, src_lengths, per, nli_data, step_type='nli_step'):
        """Feed forward process of seq2seq

        Args:
            input: Variable(LongTensor(batch_size, N)), N is the length of input sequence.
        Returns:
            list of decoded tensor
        """

        src_hidden, per_hidden, _, per_lengths = self.encode(src, src_lengths, per)
        tgt, tgt_lengths_sorted, origin_new = formalize(tgt, tgt_lengths)

        tgt_emb = self.embeddings(tgt)
        packed_tgt_emb = pack(tgt_emb, tgt_lengths_sorted.view(-1).tolist())

        tgt_output, tgt_hidden = self.tgt_rnn(packed_tgt_emb, src_hidden)
        tgt_output, _ = unpack(tgt_output)

        tgt_output = deformalize(tgt_output, origin_new)
        tgt_hidden = deformalize(tgt_hidden, origin_new)

        if step_type == 'nli_step':
            assert(nli_data is not None)
            src_hidden, tgt_hidden = self.encode_nli(nli_data)
            logits = self.project1(torch.cat([src_hidden.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
            logits = self.project2(self.dp(F.tanh(logits)))
            # logits = self.sf(logits)

        else:
            _, ref_tgt_hidden1 = self.tgt_rnn(packed_tgt_emb, per_hidden[0])
            ref_tgt_hidden1 = deformalize(ref_tgt_hidden1, origin_new)
            logits_1 = self.project1(torch.cat([ref_tgt_hidden1.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
            logits_1 = self.project2(self.dp(F.tanh(logits_1)))

            _, ref_tgt_hidden2 = self.tgt_rnn(packed_tgt_emb, per_hidden[1])
            ref_tgt_hidden2 = deformalize(ref_tgt_hidden2, origin_new)
            logits_2 = self.project1(torch.cat([ref_tgt_hidden2.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
            logits_2 = self.project2(self.dp(F.tanh(logits_2)))

            _, ref_tgt_hidden3 = self.tgt_rnn(packed_tgt_emb, per_hidden[2])
            ref_tgt_hidden3 = deformalize(ref_tgt_hidden3, origin_new)
            logits_3 = self.project1(torch.cat([ref_tgt_hidden3.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
            logits_3 = self.project2(self.dp(F.tanh(logits_3)))

            _, ref_tgt_hidden4 = self.tgt_rnn(packed_tgt_emb, per_hidden[3])
            ref_tgt_hidden4 = deformalize(ref_tgt_hidden4, origin_new)
            logits_4 = self.project1(torch.cat([ref_tgt_hidden4.squeeze(0), tgt_hidden.squeeze(0)], dim=-1))
            logits_4 = self.project2(self.dp(F.tanh(logits_4)))

            a = logits_1[:, 0].contiguous().view([-1, 1])
            b = logits_2[:, 0].contiguous().view([-1, 1])
            c = logits_3[:, 0].contiguous().view([-1, 1])
            d = logits_4[:, 0].contiguous().view([-1, 1])
            x = torch.cat([a, b, c, d], dim=1)
            y, ind = torch.max(x, dim=1)
            z = torch.cat([logits_1, logits_2, logits_3, logits_4], dim=1).view([-1, 4, 3])
            tmp = []
            for i in range(logits_1.shape[0]):
                tmp.append(z[i][ind[i]])
            logits = torch.cat(tmp, dim=0)

            # logits = (logits_1 + logits_2 + logits_3) / 3.0

            if step_type == 'self_sample':
                logits = logits.unsqueeze(0).repeat(tgt_output.size(0), 1, 1)

        return logits  # len x batch_size x 3 if step_type == 'self_sample' else batch_size x 3
