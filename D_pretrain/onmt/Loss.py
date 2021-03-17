"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
import random


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        seq_len, batch_size = batch.tgt.size()
        rewards = Variable(torch.FloatTensor([[1.0] * batch_size] * (seq_len - 1)))
        if output.is_cuda:
            rewards = rewards.cuda()
        shard_state = self._make_shard_state(batch.tgt, output, rewards, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, fak_tok, output, fak_output, nli_data, d1, d2, n1, n2, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, step_type):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note harding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)

            loss.div(normalization).backward()
            batch_stats.update(stats)
            break

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)

        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
            # self.criterion = GANLoss(self.padding_idx)
        self.confidence = 1.0 - label_smoothing
        self.d_crit = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _make_shard_state(self, fak_tok, output, rewards, range_, attns=None):
        return {
            "output": output,
            "target": fak_tok[range_[0] + 1: range_[1]],
            "rewards": rewards
        }

    def _compute_loss(self, batch, output, target, rewards):
        scores = self.generator(self._bottle(output))
        rewards = rewards.view(rewards.size(0) * rewards.size(1), 1)
        rewards.detach_()
        scores = scores * rewards.repeat(1, scores.size(1))
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            loss_data = - likelihood.sum(0)
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats

    def sharded_compute_loss(self, batch, fak_tok, output, fak_outputs, nli_data, d1, d2, n1, n2, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, step_type):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note harding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()

        # range_ = (0, batch.tgt.size(0))
        batch_size = batch.tgt.size(1)
        zero_label = Variable(torch.FloatTensor([[0.0, 1.0]]), requires_grad=False).repeat(batch_size, 1)
        zero_label = zero_label.cuda() if batch.tgt.is_cuda else zero_label
        one_label = Variable(torch.FloatTensor([[1.0, 0.0]]), requires_grad=False).repeat(batch_size, 1)
        one_label = one_label.cuda() if batch.tgt.is_cuda else one_label

        if step_type == 'd_step':
            d1_loss = self.d_crit(d1, one_label)
            d2_loss = self.d_crit(d2, zero_label)
            if random.random() > 0.5:
                d1_loss.backward()
            else:
                d2_loss.backward()

            d1_data = d1_loss.data.clone()
            d2_data = d2_loss.data.clone()
            return onmt.Statistics(d1=d1_data[0], d2=d2_data[0], step_type='d_step')

        if step_type == 'nli_step':
            src = n1
            label = nli_data[2][0].float().transpose(0, 1)
            nli_loss = self.d_crit(src, label)

            nli_loss.backward()

            _, src_ind = torch.max(src, dim=1)
            _, tgt_ind = torch.max(label, dim=1)
            acc_ind = torch.eq(src_ind, tgt_ind)
            acc_ind = acc_ind.data.float().view(1, -1)
            acc_sum = torch.sum(acc_ind)
            batch_size = float(src.shape[0])

            return onmt.Statistics(n_acc=acc_sum, n_batchsize=batch_size, step_type='nli_step')

        if step_type == 'teacher_force':
            '''
            d1.shape = len x batchsize x 1, in teacher_force mode all element is 1, torch.FloatTensor.
            '''
            rewards = d1
            range_ = (cur_trunc, cur_trunc + trunc_size)
            shard_state = self._make_shard_state(batch.tgt, output, rewards, range_, attns)
            for shard in shards(shard_state, shard_size):
                loss, stats = self._compute_loss(batch, **shard)
                (loss).div(normalization).backward()
                batch_stats.update(stats)

        if step_type == 'self_sample':
            '''
            d2.shape = len x batchsize x 1, torch.FloatTensor.
            
            n2.shape = 1 x batchsize x 1, torch.FloatTensor.
            '''
            d2 = self.sigmoid(d2)
            n2 = self.sigmoid(n2)
            '''
            print('=' * 50)
            print(d2[-1, :, :])
            print(n2[-1, :, :])
            '''

            # rewards = (d2 + n2) / 2.0
            # rewards -= 0.5
            rewards = (d2 + n2) - 0.8
            
            # print(rewards[-1, :, :])
            range_ = (0, fak_tok.size(0)+1)
            shard_state = self._make_shard_state(fak_tok, fak_outputs, rewards, range_, attns)
            for shard in shards(shard_state, shard_size):
                loss, stats = self._compute_loss(batch, **shard)
                (loss).div(normalization).backward()
                batch_stats.update(stats)
                batch_stats.step_type = 'self_sample'

        # loss, stats = self._compute_loss(batch, d_fake, **shard_state)
        # loss.div(normalization).backward()
        # batch_stats.update(stats)
        return batch_stats


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class GANLoss(torch.nn.Module):
    def __init__(self, padding_idx):
        super(GANLoss, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, prob, target, reward):
        mask = target.eq(self.padding_idx)
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss.data.masked_fill_(mask.data, 0.0)
        reward = reward.view((reward.size(0) * reward.size(1),))
        reward.detach_()
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss
