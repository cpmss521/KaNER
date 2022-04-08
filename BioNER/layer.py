# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 下午4:16
# @Author  : cp
# @File    : layer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature# scale factor
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, factor, mask=None):

        attn = torch.matmul(q*factor / self.temperature, k.transpose(2, 3))#(B,n_head,Sequence,sequence)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e30)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1,cross_attn=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        if cross_attn:
            self.w_ks = nn.Linear(2*d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(2*d_model, n_head * d_v, bias=False)
        else:
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)# math.log(n,512)*d_k**0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(-2).unsqueeze(1)&mask.unsqueeze(-2).unsqueeze(-1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, factor=math.log(len_k,512),mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        return q, attn


class Seq2KnowAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=8,d_k=64,d_v=64,dropout=0.1):
        super(Seq2KnowAttention, self).__init__()

        # self attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model,eps=1e-6)

        # cross attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads , d_k=d_k, d_v=d_v, dropout=dropout,cross_attn=True)
        self.norm2 = nn.LayerNorm(d_model,eps=1e-6)


    def forward(self, text_features, know_features, know_mask, text_mask):
        # knowledge self attention
        knowledge_out,knowledge_attn = self.self_attention(know_features,know_features,know_features,know_mask)
        knowledge_out = knowledge_out + self.dropout1(know_features)
        knowledge_out = self.norm1(knowledge_out)

        # context and knowledge attention
        dec_output,dec_attn = self.cross_attention(knowledge_out,text_features,text_features,text_mask)
        dec_output = dec_output + knowledge_out
        dec_output = self.norm2(dec_output)

        return dec_output



class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                                kernel_size=(kernel_size,),
                                padding=padding, stride=(stride,), bias=bias)
    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (B,h,S)
        x = self.conv1d(x)
        return x.transpose(1, 2)  #(B,S,h)

class Seq2QueryAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super(Seq2QueryAttention, self).__init__()

        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        init.xavier_uniform_(w4C)
        init.xavier_uniform_(w4Q)
        init.xavier_uniform_(w4mlu)

        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, c_mask, query,q_mask):
        
        score = self.trilinear_attention(context, query)  # B,S,L
        mask = c_mask.unsqueeze(-1).expand(-1, -1, q_mask.size()[1]) & q_mask.unsqueeze(-2).expand(-1, c_mask.size()[1], -1) #B*S*L
        score = score + (1.0 - mask.float()) * -1e30

        score_ = nn.Softmax(dim=2)(score)  # B,S,L
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # B,S,L
        score_t = score_t.transpose(1, 2)  # B,L,S
        c2q = torch.matmul(score_, query)  # B,S,h
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # B,S,h
        out = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)#B,S,4H
        out = self.cqa_linear(out) #B,S,H
        return out,score

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # B,S,L
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # B,S,L
        return res




class KnowledgeAttentionLayer(nn.Module):

    def __init__(self,hidden_size,prop_drop,n_heads,layer_num):
        super(KnowledgeAttentionLayer, self).__init__()

        self.context_query_atten = Seq2QueryAttention(hidden_size, prop_drop)
        self.layer_stack = nn.ModuleList([
            Seq2KnowAttention(d_model=hidden_size, n_heads=n_heads,dropout=prop_drop)
            for _ in range(layer_num)])

        self.self_attention = MultiHeadAttention(hidden_size, n_heads, d_k=64, d_v=64, dropout=prop_drop)
        self.dropout1 = nn.Dropout(prop_drop)
        self.norm1 = nn.LayerNorm(hidden_size,eps=1e-6)



    def forward(self, sequences, sequence_mask, query, query_mask,evidence,evidence_mask,
                            knowledge_embed=None, knowledge_mask=None,return_attns=False):
        """"
        sequences_batch: lstm out
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        ## context and query atttention(B,S,3H)
        att_text_query, _ = self.context_query_atten(sequences, sequence_mask, query,query_mask)
        att_text_evidence, _ = self.context_query_atten(sequences, sequence_mask, evidence,evidence_mask)
        att_text_query = torch.cat([att_text_query,att_text_evidence],dim=-1)#B*S*2H

        knowledge_out,knowledge_attn = self.self_attention(knowledge_embed,knowledge_embed,knowledge_embed,knowledge_mask)
        knowledge_out = knowledge_out + self.dropout1(knowledge_embed)
        knowledge_out = self.norm1(knowledge_out)



        ## context and knowledge attention(B,S,H)
        for dec_layer in self.layer_stack:
            knowledge_embed = dec_layer(
                att_text_query, knowledge_embed, knowledge_mask, sequence_mask)


        sequence_repr = torch.cat([sequences,att_text_query,knowledge_embed],dim=-1)#B*S*4H

        return sequence_repr



class BoundaryPointer(nn.Module):
    def __init__(self, hidden_size, inner_dim, device, RoPE=True):
        super().__init__()
        self.inner_dim = inner_dim
        self.dense = nn.Linear(4*hidden_size, self.inner_dim*2)
        self.start_outputs = nn.Linear(4*hidden_size, 1)
        self.end_outputs = nn.Linear(4*hidden_size, 1)
        self.device=device
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)# B,S,inner_dim
        return embeddings

    def forward(self, decoder_out, sequence_mask):

        batch_size,seq_len,_ = decoder_out.size()

        start_logits = self.start_outputs(decoder_out).squeeze(-1)#B*S
        end_logits = self.end_outputs(decoder_out).squeeze(-1)#B*S

        outputs = self.dense(decoder_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:

            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        span_logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        span_logits = span_logits.squeeze(1)#B*S*S

        # padding mask
        pad_mask = sequence_mask.unsqueeze(-1).expand(-1, -1, seq_len) & sequence_mask.unsqueeze(-2).expand(-1, seq_len, -1)
        span_logits = mask_logits(span_logits,pad_mask)
        #upright triangle
        mask = torch.tril(torch.ones_like(span_logits), -1)
        span_logits = span_logits - mask * 1e30

        span_logits = (span_logits / self.inner_dim ** 0.5)#B*S*S

        return start_logits,end_logits,span_logits

