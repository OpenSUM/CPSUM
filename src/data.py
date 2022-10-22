import os
from functools import partial
import config

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from transformers import BertTokenizer

from utils import read_jsonl, write_jsonl, append_jsonl, get_logger

log = get_logger(__name__)


class CPSUMDataModule:
    def __init__(
            self,
            dataset_name: str,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            tokenizer_name_or_path: str,
            train_dataset_name: str,
            val_dataset_name: str,
            test_dataset_name: str,
            unsupervised_dataset_name: str,
            num_workers: int = 0,
            src_max_len: int = 512,
            max_text_ntokens_per_sent: int = 200,
            min_text_ntokens_per_sent: int = 5,
    ):
        self.dataset_name = dataset_name
        self.train_filename = os.path.join("./data", dataset_name, train_dataset_name)
        self.val_filename = os.path.join("./data", dataset_name, val_dataset_name)
        self.test_filename = os.path.join("./data", dataset_name, test_dataset_name)
        self.unsupervised_train_filename = os.path.join("./data", dataset_name, unsupervised_dataset_name)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)

        self.dataset_info = {
            "text_max_len": src_max_len,
            "max_text_ntokens_per_sent": max_text_ntokens_per_sent,
            "min_text_ntokens_per_sent": min_text_ntokens_per_sent,
        }
        self.text_max_len = self.dataset_info["text_max_len"]
        self.max_text_ntokens_per_sent = self.dataset_info["max_text_ntokens_per_sent"]
        self.min_text_ntokens_per_sent = self.dataset_info["min_text_ntokens_per_sent"]

        self.collator = partial(collator,
                                cls_id=self.tokenizer.cls_token_id,
                                sep_id=self.tokenizer.sep_token_id,
                                pad_id=self.tokenizer.pad_token_id,
                                )

    def prepare(self, do_consist_training, supervised_dataset_size, supervised_filename):
        text_args = {
            "text_max_len": self.text_max_len,
            "max_text_ntokens_per_sent": self.max_text_ntokens_per_sent,
            "min_text_ntokens_per_sent": self.min_text_ntokens_per_sent,
        }

        data = read_jsonl(self.train_filename, l=supervised_dataset_size)
        self.train_filename = os.path.join("./data", self.dataset_name, supervised_filename)
        write_jsonl(self.train_filename, data)

        self.train_dataset = CPSUMDataset(self.train_filename, self.tokenizer, self.tokenizer_name_or_path, **text_args)
        self.val_dataset = CPSUMDataset(self.val_filename, self.tokenizer, self.tokenizer_name_or_path, **text_args)
        self.test_dataset = CPSUMDataset(self.test_filename, self.tokenizer, self.tokenizer_name_or_path, **text_args)

        if do_consist_training:
            self.unsupervised_train_dataset = CPSUMDataset(self.unsupervised_train_filename, self.tokenizer, self.tokenizer_name_or_path, **text_args)

        # if do_consist_training:
        #     self.unsupervised_train_dataset = Subset(self.unsupervised_train_dataset, range(unsupervised_dataset_size))

        log.info(f"train dataset: {supervised_dataset_size} for supervised.")
        log.info(f"train_batch_size: {self.train_batch_size}")

    def append_pseudo(self, pseudo_data, supervised_filename):
        text_args = {
            "text_max_len": self.text_max_len,
            "max_text_ntokens_per_sent": self.max_text_ntokens_per_sent,
            "min_text_ntokens_per_sent": self.min_text_ntokens_per_sent,
        }

        self.train_filename = os.path.join("./data", self.dataset_name, supervised_filename)
        append_jsonl(self.train_filename, pseudo_data)
        self.train_dataset = CPSUMDataset(self.train_filename, self.tokenizer, **text_args)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collator,
            drop_last=True,
        )

    def unsupervised_dataloader(self):
        assert getattr(self,
                       "unsupervised_train_dataset") is not None, \
            "Unsupervised training dataset is not properly loaded!"

        return DataLoader(
            dataset=self.unsupervised_train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collator,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collator,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collator,
            drop_last=True,
        )


class CPSUMDataset(Dataset):
    def __init__(self,
                 data_filename,
                 tokenizer,
                 tokenizer_name_or_path,
                 text_max_len: int,
                 max_text_ntokens_per_sent: int,
                 min_text_ntokens_per_sent: int,
                 ):
        self.data_filename = data_filename
        self.data = read_jsonl(data_filename)

        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.max_text_ntokens_per_sent = max_text_ntokens_per_sent
        self.min_text_ntokens_per_sent = min_text_ntokens_per_sent

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = self.data[index]
        text, labels, golden_tgt = ex["src"], ex["labels"], ex["golden_tgt"]
        labels = [float(i) for i in labels]
        _raw_labels = labels
        _text_ids, _truncated_labels, _text = self.text2id(text, labels)

        _text_ids = pad_1d(_text_ids, self.text_max_len, self.tokenizer.pad_token_id)
        _segments_ids = self.textid2seg(_text_ids)
        _text_ids = torch.unsqueeze(_text_ids, 0)
        _segments_ids = torch.unsqueeze(_segments_ids, 0)

        if "unsupervised" in self.data_filename:
            choice_aug = 1
            for i in range(choice_aug):
                text_ids, _, text = self.text2id(ex["augmented"][i], labels)

                text_ids = pad_1d(text_ids, self.text_max_len, self.tokenizer.pad_token_id)
                segments_ids = self.textid2seg(text_ids)
                text_ids = torch.unsqueeze(text_ids, 0)
                segments_ids = torch.unsqueeze(segments_ids, 0)
                
                _text_ids = torch.cat([_text_ids, text_ids])
                _segments_ids = torch.cat([_segments_ids, segments_ids])

        return _text_ids, _segments_ids, _truncated_labels, _text, _raw_labels, golden_tgt

    def textid2seg(self, textids):
        tmp = []
        _segs = [-1] + [i for i, t in enumerate(textids) if t == self.tokenizer.pad_token_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                tmp += s * [0]
            else:
                tmp += s * [1]
        if len(tmp) < len(textids):
            tmp += (len(textids) - len(tmp)) * [0]
        tmp = torch.tensor(tmp)
        return tmp

    def text2id(self, text, labels):
        pseudo = 0
        if len(labels) < len(text):
            pseudo = 1
        
        add_special_token_text = list(map(lambda sent: self.tokenizer.cls_token + sent, text))
        subtokens = list(map(self.tokenizer.tokenize, add_special_token_text))

        mask_idxs = [i for i, t in enumerate(subtokens) if len(t) > self.min_text_ntokens_per_sent]

        subtokens = [subtokens[idx] for idx in mask_idxs]
        text = [text[idx] for idx in mask_idxs]
        if pseudo == 0:
            labels = [labels[idx] for idx in mask_idxs]

        sent_ids = list(map(self.tokenizer.convert_tokens_to_ids, subtokens))
        sent_ids = [ids[:self.max_text_ntokens_per_sent - 1] + [self.tokenizer.sep_token_id] for ids in sent_ids]

        text_ids = list()
        for i in range(len(sent_ids)):
            if len(text_ids) + len(sent_ids[i]) <= self.text_max_len:
                text_ids.extend(sent_ids[i])
            else:
                remain_len = self.text_max_len - len(text_ids)
                if remain_len > self.min_text_ntokens_per_sent:
                    text_ids.extend(sent_ids[i][:remain_len - 1] + [self.tokenizer.sep_token_id])
                break

        text_ids = torch.tensor(text_ids)

        sent_num = (text_ids == self.tokenizer.cls_token_id).sum()

        sent_ids = sent_ids[:sent_num]
        if pseudo == 0:
            labels = labels[:sent_num]

        return text_ids, labels, text

# =========================================================COLLATOR=============================================================


class Batch:
    def __init__(
            self,
            input_ids,
            attn_mask,
            cls_mask,
            sep_mask,
            seg,
            raw_labels,
            sep_labels,
            texts,
            labels,
            golden_tgt,
    ):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.cls_mask = cls_mask
        self.sep_mask = sep_mask
        self.seg = seg
        self.raw_labels = raw_labels
        self.sep_labels = sep_labels
        self.texts = texts
        self.labels = labels
        self.golden_tgt = golden_tgt

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attn_mask = self.attn_mask.to(device)
        self.cls_mask = self.cls_mask.to(device)
        self.sep_mask = self.sep_mask.to(device)
        self.seg = self.seg.to(device)
        self.labels = self.labels.to(device)

        return self

    def __len__(self):
        return self.input_ids.size(0)


def pad_1d(x, pad_len, pad_id):
    xlen = x.size(0)
    if xlen < pad_len:
        new_x = x.new_empty([pad_len], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen] = x
        x = new_x
    elif xlen > pad_len:
        end_id = x[-1]
        x = x[:pad_len]
        x[-1] = end_id
    return x


def pad_2d(x, pad_len, pad_id):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < pad_len:
        new_x = x.new_zeros([pad_len, xdim], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen, :] = x
        x = new_x
    return x


def collator(items, cls_id, sep_id, pad_id):
    input_ids, segments_ids, labels, texts, raw_labels, golden_tgt = zip(*items)
    input_ids = torch.stack([ids.clone().detach() for ids in input_ids])
    segments_ids = torch.stack([ids.clone().detach() for ids in segments_ids])

    attn_mask = ~(input_ids == pad_id)

    cls_mask = input_ids == cls_id
    sep_mask = input_ids == sep_id

    merge_labels = torch.tensor(sum(labels, list()))

    return Batch(
        input_ids=input_ids,        # source text2ids (for model input)
        attn_mask=attn_mask,
        cls_mask=cls_mask,
        sep_mask=sep_mask,
        seg=segments_ids,
        raw_labels=raw_labels,      # labels before truncating
        sep_labels=labels,          # labels after truncating
        texts=texts,                # source texts
        labels=merge_labels,        # Merged truncated labels in each batch (for loss calculation)
        golden_tgt=golden_tgt       # source summaries (for evaluation)
    )


