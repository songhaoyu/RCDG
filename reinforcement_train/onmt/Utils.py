import torch
from torch.autograd import Variable


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def formalize(batch, batch_length, batch_first=False):
    """formalize a batch to sort the batch according to its length

    Args:
        batch: batch
        batch_length: batch length list
    Returns:
        formalized batch
    """
    sorted_lengths, _ = torch.sort(batch_length, descending=True)
    batch_length = batch_length.view(-1).tolist()
    index_length = [(i, l) for i, l in enumerate(batch_length)]
    ordered_index = sorted(index_length, key=lambda e: e[1], reverse=True)

    origin_new = dict([(v[0], k) for k, v in enumerate(ordered_index)])

    sorted_batch = Variable(batch.data.new(batch.size()))
    for k, v in origin_new.items():
        if batch_first:
            sorted_batch[v] = batch[k]
        else:
            sorted_batch[:, v] = batch[:, k]
    return sorted_batch, sorted_lengths, origin_new


def deformalize(batch, origin_new):
    """reform batch in the origin order, batch is the second dimension.

    Args:
        batch: encoded batch, length*batch_size*dim
        origin_new: origin->new index dict
    Returns:
        reformed batch
    """
    desorted_batch = Variable(batch.data.new(batch.size()))
    for k, v in origin_new.items():
        desorted_batch[:, k] = batch[:, v]
    return desorted_batch