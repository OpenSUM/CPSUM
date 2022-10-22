import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import logging
import config
from utils import get_logger
import math

log = get_logger(__name__)


class CPSUM(nn.Module):
    def __init__(self, encoder_name_or_path: str, args):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name_or_path)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.hidden_dropout_prob = 0.1
        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.hidden_size = args.hidden_size
        self.PLform = args.PLform

    def forward(self, batch):
        input_ids, attn_mask = batch.input_ids, batch.attn_mask
        cls_mask, sep_mask, seg = batch.cls_mask, batch.sep_mask, batch.seg
        label_len = []
        for i in range(len(batch)):
            label_len.append(len(batch.sep_labels[i]))

        for i in range(input_ids.size(1)):
            enc_outputs = self.encoder(input_ids=input_ids[:, i, :],
                                       attention_mask=attn_mask[:, i, :],
                                       token_type_ids=seg[:, i, :],
                                       )
            enc_token_embs = enc_outputs.last_hidden_state

            # 取出输出中所有cls字符的向量
            tmp_cls_mask = cls_mask[:, i, :].unsqueeze(-1)
            flatten_enc_sent_embs = enc_token_embs.masked_select(tmp_cls_mask)
            enc_sent_embs = flatten_enc_sent_embs.view(-1, enc_token_embs.size(-1))
            logits = self.head(enc_sent_embs)
            logits = logits.view(-1)
            logits = torch.sigmoid(logits)

            if self.PLform != 'hard':
                hard_PLs = []
            if i == 0 and self.PLform == 'hard':
                hard_PLs = []
                pos = 0
                inf = -99999

                for k in range(len(label_len)):
                    max_number = []
                    max_index = []
                    logits_k = logits[pos:pos + label_len[k]].cpu().detach().numpy().tolist()
                    if len(logits_k) < 3:
                        logits_k = [1.0 for i in range(len(logits_k))]
                    else:
                        for _ in range(1):
                            number = max(logits_k)
                            index = logits_k.index(number)
                            logits_k[index] = inf
                            max_number.append(number)
                            max_index.append(index)
                        logits_k = [(1.0 if i in max_index else 0.0) for i in range(len(logits_k))]

                    hard_PLs.append(logits_k)
                    pos = pos + label_len[k]

            if i == 0:
                pseudo_labels = []
                pos = 0
                for k in range(len(label_len)):
                    pseudo_labels.append(F.softmax(logits[pos:pos+label_len[k]], dim=0).cpu().detach().numpy().tolist())
                    pos = pos + label_len[k]

            logits = torch.unsqueeze(logits, 0)
            if i == 0:
                _logits = logits
                dim = logits.size(1)
            else:
                logits = F.pad(logits, [0, dim - logits.size(1)])
                _logits = torch.cat([_logits, logits])

        return _logits, pseudo_labels, hard_PLs


