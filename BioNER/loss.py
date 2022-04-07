# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 下午8:05
# @Author  : cp
# @File    : loss.py

from abc import ABC
import torch
from torch.nn.modules import BCEWithLogitsLoss
from BioNER.util import extract_ner_spans

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class KaNERLoss(Loss):
    def __init__(self,model, optimizer, scheduler, max_grad_norm,weight_start,weight_end,weight_span):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.weight_span = weight_span

    def compute(self,start_logits, end_logits, span_logits,
                     start_labels, end_labels, span_labels,sequence_mask):

        batch_size, seq_len,_ = span_labels.size()
        start_loss = self.categorical_loss(start_logits, start_labels.float())
        end_loss = self.categorical_loss(end_logits, end_labels.float())

        span_loss = self.categorical_loss(span_logits.view(batch_size, -1), span_labels.view(batch_size, -1).float())
        total_loss = self.weight_start*start_loss + self.weight_end*end_loss + self.weight_span*span_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()

        return total_loss.item()


    def categorical_loss(self,y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()
