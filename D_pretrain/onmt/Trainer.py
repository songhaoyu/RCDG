from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from torch.autograd import Variable


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0, d1=0, d2=0, n_acc=0, n_batchsize=0, step_type='teacher_force'):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.d1 = d1
        self.d2 = d2
        self.n_acc = n_acc
        self.n_batch = n_batchsize
        self.step_type = step_type
        self.num_batchs = 0

    def update(self, stat, mode='train'):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_acc += stat.n_acc
        self.n_batch += stat.n_batch
        self.step_type = stat.step_type


        if mode == 'valid':
            self.d1 += math.exp(-stat.d1)
            self.d2 += math.exp(-stat.d2)
            self.num_batchs += 1
        else:
            self.d1 = math.exp(-stat.d1)
            self.d2 = math.exp(-stat.d2)
            self.num_batchs = 1

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        self.n_words += 1
        return math.exp(min(self.loss / self.n_words, 100))

    def d_acc(self):
        return (self.d1 / self.num_batchs + self.d2 / self.num_batchs) / 2

    def nli_acc(self):
        return self.n_acc / self.n_batch

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        if self.step_type == 'teacher_force' or self.step_type == 'self_sample':
            print(("Epoch %2d, %s %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                   "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                  (epoch, self.step_type, batch,  n_batches,
                   self.accuracy(),
                   self.ppl(),
                   self.n_src_words / (t + 1e-5),
                   self.n_words / (t + 1e-5),
                   time.time() - start))
        elif self.step_type == 'd_step':
            print("Epoch %2d, %s %5d/%5d; d: %.5f" %
                  (epoch, self.step_type, batch, n_batches, self.d_acc()))
        elif self.step_type == 'nli_step':
            print("Epoch %2d, %s %5d/%5d; nli: %.5f" %
                  (epoch, self.step_type, batch, n_batches, self.nli_acc()))

        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model, disc, nli,
                 train_loss, valid_loss, g_optim, d_optim, nli_optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 normalization="sents", accum_count=1):
        # Basic attributes.
        self.model = model
        self.disc = disc
        self.nli = nli
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.nli_optim = nli_optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.accum_count = accum_count
        self.padding_idx = self.train_loss.padding_idx
        self.eos_idx = self.train_loss.tgt_vocab.stoi[onmt.io.EOS_WORD]
        self.bos_idx = self.train_loss.tgt_vocab.stoi[onmt.io.BOS_WORD]
        self.normalization = normalization
        assert(accum_count > 0)
        if accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        report_freq = 40
        idx = 0
        truebatch = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        def gradient_accumulation(truebatch_, total_stats_,
                                  report_stats_, nt_, step_type_):
            if self.accum_count > 1:
                self.model.zero_grad()
                self.disc.zero_grad()
                self.nli.zero_grad()

            # d1_acc, d2_acc = 0.0, 0.0

            for batch in truebatch_:
                target_size = batch.tgt.size(0)
                # Truncated BPTT
                if self.trunc_size:
                    trunc_size = self.trunc_size
                else:
                    trunc_size = target_size

                dec_state = None
                src = onmt.io.make_features(batch, 'src', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                    per = batch.per
                    nli_data = batch.nli

                    '''
                    print('-'*50)
                    for n, n_len in nli_data:
                        print(n)
                        print(n_len)
                    '''

                    report_stats.n_src_words += src_lengths.sum()
                else:
                    src_lengths = None
                    ref_lengths = None

                tgt_outer = onmt.io.make_features(batch, 'tgt')

                for j in range(0, target_size-1, trunc_size):
                    # 1. Create truncated target.
                    tgt = tgt_outer[j: j + trunc_size]
                    tgt_lengths = self.get_length(tgt[1:])  # exclude start symbol

                    # 2. F-prop all but generator.
                    if self.accum_count == 1:
                        self.model.zero_grad()
                        self.disc.zero_grad()
                        self.nli.zero_grad()

                    outputs, attns, fak_tok, fak_outputs, d1, d2, n1, n2 = None, None, None, None, None, None, None, None

                    if step_type_ == 'self_sample':
                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        # ref_context, ref_lengths = self.model.encode_ref(ref)
                        ref_context, ref_lengths = None, None
                        fak_tok, fak_outputs = self.model.infer(ref_context, ref_lengths, tgt.size(0) - 1, enc_state,
                                                                context, src_lengths, self.bos_idx, sample=False)
                        fak_tok = self.mask_eos(fak_tok)
                        fak_tok_lengths = self.get_length(fak_tok[1:])  # exclude start symbol
                        fak_tok = fak_tok[: torch.max(fak_tok_lengths)+1]  # remove extra padding symbols in tail
                        fak_outputs = fak_outputs[: torch.max(fak_tok_lengths)]

                        # d2 = self.roll_out(src, src_lengths, fak_tok[1:], fak_tok_lengths, per, 3, enc_state, context, src_lengths)[:, :, :1]
                        d2 = self.disc(fak_tok[1:], fak_tok_lengths, src, src_lengths, per, step_type='self_sample')[:, :, :1]
                        # n2 = self.nli(fak_tok[1:], fak_tok_lengths, src, src_lengths, per, nli_data, step_type='self_sample')[:, :, 2:3]

                    if step_type_ == 'teacher_force':
                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        outputs, dec_state, attns = self.model.decoder(tgt[:-1], context,
                                                                       enc_state if dec_state is None
                                                                       else dec_state,
                                                                       context_lengths=src_lengths)

                        d1 = Variable(torch.ones(tgt[1:].size()), requires_grad=False)
                        n1 = Variable(torch.ones(tgt[1:].size()), requires_grad=False)
                        if tgt.is_cuda:
                            d1 = d1.cuda()
                            n1 = n1.cuda()

                    if step_type_ == 'd_step':
                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        ref_context, ref_lengths = None, None
                        fak_tok, fak_outputs = self.model.infer(ref_context, ref_lengths, tgt.size(0)-1, enc_state, context, src_lengths, self.bos_idx)
                        fak_tok = self.mask_eos(fak_tok)
                        fak_tok_lengths = self.get_length(fak_tok[1:])  # exclude start symbol

                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        outputs, dec_state, attns = self.model.decoder(tgt[:-1], context,
                                                                       enc_state if dec_state is None
                                                                       else dec_state,
                                                                       context_lengths=src_lengths)

                        d1 = self.disc(tgt[1:], tgt_lengths, src, src_lengths, per)
                        d2 = self.disc(fak_tok[1:], fak_tok_lengths, src, src_lengths, per)

                    if step_type_ == 'nli_step':
                        n1 = self.nli(tgt[1:], tgt_lengths, src, src_lengths, per, nli_data, step_type=step_type)
                        n2 = n1

                    # 3. Compute loss in shards for memory efficiency.
                    batch_stats = self.train_loss.sharded_compute_loss(
                            batch, fak_tok, outputs, fak_outputs, nli_data, d1, d2, n1, n2, attns, j,
                            trunc_size, self.shard_size, nt_, step_type)

                    # 4. Update the parameters and statistics.
                    if self.accum_count == 1:
                        if step_type_ == 'd_step':
                            self.d_optim.step()
                        elif step_type_ == 'nli_step':
                            pass
                            # self.nli_optim.step()
                        else:
                            pass
                            # self.g_optim.step()
                    total_stats_.update(batch_stats)
                    report_stats_.update(batch_stats)

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()

            if self.accum_count > 1:
                if step_type_ == 'd_step':
                    self.d_optim.step()
                elif step_type == 'nli_step':
                    pass
                    # self.nli_optim.step()
                else:
                    pass
                    # self.g_optim.step()

            # if d1_acc < 0.5 or (1-d2_acc) < 0.5:
            #     step_type_ = 'd_step'
            # else:
            #     step_type_ = 'g_step'
            # return step_type_

        step_type = 'd_step'
        for i, batch_ in enumerate(train_iter):
            if i % report_freq < 10:
                step_type = 'd_step'
            elif i % report_freq < 20:
                step_type = 'd_step'
            elif i % report_freq < 30:
                step_type = 'd_step'
            else:
                step_type = 'd_step' 

            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            truebatch.append(batch_)
            accum += 1
            if self.normalization is "tokens":
                normalization += batch_.tgt[1:].data.view(-1) \
                                       .ne(self.padding_idx)
            else:
                normalization += batch_.batch_size

            if accum == self.accum_count:
                gradient_accumulation(
                        truebatch, total_stats,
                        report_stats, normalization, step_type)

                if report_func is not None:
                    if i % 10 == 0:
                        report_flag = True
                    else:
                        report_flag = False

                    report_stats = report_func(
                        epoch, idx, num_batches,
                        total_stats.start_time, self.g_optim.lr,
                        report_stats, report_flag)

                truebatch = []
                accum = 0
                normalization = 0
                idx += 1

        if len(truebatch) > 0:
            gradient_accumulation(
                    truebatch, total_stats,
                    report_stats, normalization)
            truebatch = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')
            per = batch.per

            # F-prop through the model.
            outputs, attns, _, _ = self.model(src, tgt, per, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        self.d_optim.update_learning_rate(ppl, epoch)
        return self.g_optim.update_learning_rate(ppl, epoch)
        #pass

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        disc_state_dict = self.disc.state_dict()
        nli_state_dict = self.nli.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'disc': disc_state_dict,
            'nli': nli_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'g_optim': self.g_optim,
            'd_optim': self.d_optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    # def mask_eos(self, v):
    #     vdata = v.data.clone()
    #     mask = vdata.le(self.eos_idx)
    #     return Variable(vdata.masked_fill_(mask, 0))

    def roll_out(self, src, src_lengths, seq, seq_lengths, ref, roll_num, init_state, context, context_lengths):
        src_hidden, ref_hidden, _, _ = self.disc.encode(src, src_lengths, ref)
        seq_len, batch_size, _ = seq.size()
        rewards = []
        for i in range(roll_num):
            for l in range(1, seq_len):
                data = seq[0: l]  # l from 1 to seq_len-1
                samples = self.model.sample(
                    batch_size, seq_len, data, init_state, context, context_lengths, self.bos_idx)
                sample_lengths = self.get_length(samples)
                pred = self.disc(samples, sample_lengths, src, src_lengths, ref, step_type='d_step').unsqueeze(0)
                if i == 0:
                    rewards.append(pred.data)
                else:
                    rewards[l-1] += pred.data

            pred = self.disc(seq, seq_lengths, src, src_lengths, ref, step_type='d_step').unsqueeze(0)
            if i == 0:
                rewards.append(pred.data)
            else:
                rewards[seq_len-1] += pred.data

        rewards = Variable(torch.cat(rewards, dim=0) / (1.0 * roll_num))
        '''
        print('='*50)
        print(rewards)
        '''
        return rewards

    def mask_eos(self, tensor):
        t = tensor.squeeze(2).t()
        batch_size, seq_len = t.size()
        for i in range(batch_size):
            flag = False
            for j in range(seq_len):
                if flag:
                    t[i][j].data[0] = self.padding_idx
                elif t[i][j].data[0] == self.eos_idx:
                    flag = True
        return t.t().unsqueeze(2)

    def get_length(self, x):
        x = x.squeeze(2).t()
        mask = x.ne(self.padding_idx).long()
        return torch.sum(mask, dim=1).data

    def decode_tensor(self, t):
        t = t.squeeze(2).t()
        for sample in t:
            print(' '.join([self.train_loss.tgt_vocab.itos[i.data[0]] for i in sample]))
