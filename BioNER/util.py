# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午5:50
# @Author  : cp
# @File    : util.py.py


import csv
import json
import os
import math
import random
import numpy as np
import torch


CSV_DELIMETER = ';'


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'

    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed):
    """ set random seed for numpy and torch, more information here:
        https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed: the random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)



def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked




def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        if key=='query_len':
            converted_batch[key] = torch.LongTensor(batch[key]).to(device)
        elif key =='evidence_len':
            converted_batch[key] = torch.LongTensor(batch[key]).to(device)
        else:
            converted_batch[key] = batch[key].to(device)

    return converted_batch



def extract_ner_spans(start_pred, end_pred,mask):
    '''
    :param start_logit: (b, s)
    :param end_logit: (b, s)
    :return:
    '''
    batch_size,sequence = start_pred.size()
    pred_spans = []
    lens = mask.sum(dim=1).tolist()
    for start_ids, end_ids, l in zip(start_pred, end_pred, lens):

        span = torch.zeros([sequence,sequence],dtype=torch.bool).to(start_pred.device)
        for i, s in enumerate(start_ids[:l]):
            if s == 0:
                continue
            for j, e in enumerate(end_ids[i:l]):
                if (s == e) and (j)<=7:
                    span[i,i+j] =1
                # if s == e:
                #     span[i,i+j] =1
                #     break
        pred_spans.append(span)
    pred_spans = torch.stack(pred_spans)
    return pred_spans


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        refer to: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
    """

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
