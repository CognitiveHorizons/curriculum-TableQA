import math
import random
from typing import Tuple

import torch
import numpy as np


def weighted_choice(objects, weights):
    """
    returns randomly an element from the sequence of 'objects',
        the likelihood of the objects is weighted according
        to the sequence of 'weights', i.e. percentages.
    """
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


def weighted_sample(objects, weights, k):
    """
    This function draws a random sample (without repeats)
    of length k     from the sequence 'objects' according
    to the list of weights
    """
    sample = set()
    objects = list(objects)
    weights = list(weights)
    while len(sample) < k:
        choice = weighted_choice(objects, weights)
        sample.add(choice)
        index = objects.index(choice)
        weights.pop(index)
        objects.remove(choice)
        weights = [ x / sum(weights) for x in weights]
    return list(sample)


# Shamelessly copied from `https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py`
def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ValueError("Both the tensor and sequence lengths must be torch.Tensors.")
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def init_random_seed(seed, device: torch.device):
    random.seed(seed)
    torch.manual_seed(seed * 29 // 7)
    np.random.seed(seed * 13 // 7)
    # create agent
    if 'cuda' in device.type:
        torch.cuda.manual_seed(seed * 29 // 7)

def to_padded_tensor(val_list, pad_id=0, dtype=np.int64):
    batch_size = len(val_list)
    max_entry_num = max(len(x) for x in val_list)

    tensor = np.zeros((batch_size, max_entry_num), dtype=dtype)
    tensor.fill(pad_id)
    for i, val in enumerate(val_list):
        tensor[i, :len(val)] = val

    return torch.from_numpy(tensor)


def get_sequence_mask_from_length_array(question_lengths):
    seq_mask = torch.zeros(len(question_lengths), max(question_lengths), dtype=torch.float)
    for e_id, src_len in enumerate(question_lengths):
        seq_mask[e_id, :src_len] = 1

    return seq_mask


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            torch.nn.init.xavier_normal(p.data)


def dot_prod_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    entry_masks: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

    if entry_masks is not None:
        att_weight.data.masked_fill_((1.0 - entry_masks).bool(), -float('inf'))

    att_prob = torch.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_prob.view(*att_view), values).squeeze(1)

    return ctx_vec, att_prob


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        yield examples


def loop_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    while True:
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]

            yield examples



def batch_iter_dds(data, batch_size, shuffle=False):
    """
    Takes in a the list of allotted environments and returns a batch to be pushed 
    into the queue.

    Args:
        data: list of envs
        batch_size: -
        shuffle: -
    """
    
    batch_num = math.ceil(len(data) / batch_size)
    real_data_list = []
    syn_data_list =[]
    data_dict = {}
    data_dict["select-wc1"]=[]
    data_dict["select-wc2"]=[]
    data_dict["select-wc3"]=[]
    data_dict["select-wc4"]=[]
    data_dict["sum-wc1"]=[]
    data_dict["sum-wc2"]=[]
    data_dict["sum-wc3"]=[]
    data_dict["sum-wc4"]=[]
    data_dict["avg-wc1"]=[]
    data_dict["avg-wc2"]=[]
    data_dict["avg-wc3"]=[]
    data_dict["avg-wc4"]=[]
    data_dict["max-wc1"]=[]
    data_dict["max-wc2"]=[]
    data_dict["max-wc3"]=[]
    data_dict["max-wc4"]=[]
    data_dict["min-wc1"]=[]
    data_dict["min-wc2"]=[]
    data_dict["min-wc3"]=[]
    data_dict["min-wc4"]=[]
    category_list = list(data_dict.keys())
    for d in data:
        if d.d_type=="real":
            real_data_list.append(d)
        elif d.d_type =="syn":
            syn_data_list.append(d)
            if d.op =="select":
                if d.qtype == "1wc":
                    data_dict["select-wc1"].append(d)
                if d.qtype == "2wc":
                    data_dict["select-wc2"].append(d)
                if d.qtype == "3wc":
                    data_dict["select-wc3"].append(d)
                if d.qtype == "4wc":
                    data_dict["select-wc4"].append(d)
            if d.op == "sum":
                if d.qtype == "1wc":
                    data_dict["sum-wc1"].append(d)
                if d.qtype == "2wc":
                    data_dict["sum-wc2"].append(d)
                if d.qtype == "3wc":
                    data_dict["sum-wc3"].append(d)
                if d.qtype == "4wc":
                    data_dict["sum-wc4"].append(d)
            if d.op == "average":
                if d.qtype == "1wc":
                    data_dict["avg-wc1"].append(d)
                if d.qtype == "2wc":
                    data_dict["avg-wc2"].append(d)
                if d.qtype == "3wc":
                    data_dict["avg-wc3"].append(d)
                if d.qtype == "4wc":
                    data_dict["avg-wc4"].append(d)
            if d.op == "minimum":
                if d.qtype == "1wc":
                    data_dict["min-wc1"].append(d)
                if d.qtype == "2wc":
                    data_dict["min-wc2"].append(d)
                if d.qtype == "3wc":
                    data_dict["min-wc3"].append(d)
                if d.qtype == "4wc":
                    data_dict["min-wc4"].append(d)
            if d.op == "maximum":
                if d.qtype == "1wc":
                    data_dict["max-wc1"].append(d)
                if d.qtype == "2wc":
                    data_dict["max-wc2"].append(d)
                if d.qtype == "3wc":
                    data_dict["max-wc3"].append(d)
                if d.qtype == "4wc":
                    data_dict["max-wc4"].append(d)
    
    syn_length = len(syn_data_list)
    
    if syn_length == 0:
        index_array = list(range(len(data)))
    else:
        real_index_array = list(range(len(real_data_list)))
        syn_index_array = list(range(len(syn_data_list)))
    if shuffle:
        if syn_length ==0:
            np.random.shuffle(index_array)
        else:
            np.random.shuffle(real_index_array)
            np.random.shuffle(syn_index_array)
    

    # real index array -> array of indices in data that are real examples

    for i in range(batch_num):
        if syn_length != 0:
            # ??

            # 70 percent examples are real.
            real_batch_size = int(batch_size*0.7)
            # syn_batch_size = int(batch_size*1.0)
            # real_indices = real_index_array[i * real_batch_size: (i + 1) * real_batch_size]
            # syn_indices = syn_index_array[i * syn_batch_size: (i + 1) * syn_batch_size]
            # indices = real_indices+syn_indices
            real_indices = real_index_array[i * real_batch_size: (i + 1) * real_batch_size]
            #print(batch_size)
            #choices = random.choices([0,1,2,3,4,5,6,7,8,9,10], cum_weights=(60, 10, 30,30,30,30,10,40,40,40,30), k=batch_size)
            op_choices = list(np.random.choice(['select','sum','avg','min','max'], batch_size*2, p=[0.2,0.4,0.2,0.1,0.1]))
            wc_choices = list(np.random.choice(['wc1','wc2','wc3','wc4'], batch_size*2, p=[0.1,0.4,0.3,0.2]))
            #print(op_choices)
            #print(wc_choices)
            choices =[]
            data_to_return =[]
            for op,wc in zip(op_choices,wc_choices):
                choices.append(op+"-"+wc)
            added =0
            for c in choices:
                #print(len(data_dict[c]))
                if len(data_dict[c])==0:
                    continue
                else:
                    #print(random.sample(data_dict[c],1))
                    data_to_return.append(random.sample(data_dict[c], 1)[0])
                    added+=1
                if added==int(batch_size*0.3):
                    break
        else:
            indices = index_array[i * batch_size: (i + 1) * batch_size]
        
        if syn_length ==0:
            examples = [data[idx] for idx in indices]
        else:
            examples = [real_data_list[idx] for idx in real_indices]+data_to_return
            #print(examples)
        yield examples


# def batch_iter(data, batch_size, shuffle=False):
#     batch_num = math.ceil(len(data) / batch_size)
#     real_data_list = []
#     syn_data_list =[]
#     for d in data:
#         if d.d_type=="real":
#             real_data_list.append(d)
#         elif d.d_type =="syn":
#             syn_data_list.append(d)
#     syn_length = len(syn_data_list)
#     if syn_length == 0:
#         index_array = list(range(len(data)))
#     else:
#         real_index_array = list(range(len(real_data_list)))
#         syn_index_array = list(range(len(syn_data_list)))
#     if shuffle:
#         if syn_length ==0:
#             np.random.shuffle(index_array)
#         else:
#             np.random.shuffle(real_index_array)
#             np.random.shuffle(syn_index_array)
#     for i in range(batch_num):
#         if syn_length != 0:
#             real_batch_size = int(batch_size*9.0)
#             syn_batch_size = int(batch_size*1.0)
#             real_indices = real_index_array[i * real_batch_size: (i + 1) * real_batch_size]
#             syn_indices = syn_index_array[i * syn_batch_size: (i + 1) * syn_batch_size]
#             indices = real_indices+syn_indices
#         else:
#             indices = index_array[i * batch_size: (i + 1) * batch_size]
#         examples = [data[idx] for idx in indices]
#         yield examples