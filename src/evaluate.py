import re

import torch
import torch.nn.functional as F

from model import CPSUM
from utils import get_logger
from metrics import (
    calc_rouge_from_pyrouge,
    calc_rouge_from_python_implementation
)


log = get_logger(__name__)


def extract_oracle_from_logits(logits, texts, summaries, extract_n_sents, trigram_block=True):
    def text_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def get_ngrams(n, text):
        if isinstance(text, str):
            text = text_clean(text)
            text = text.split()

        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def has_same_trigram(c, p):
        tri_c = get_ngrams(3, text_clean(c).split())

        if isinstance(p, str):
            p = [p]

        for i in range(len(p)):
            tri_p = get_ngrams(3, text_clean(p[i]).split())
            if len(tri_c.intersection(tri_p)) > 0:
                return True

        return False

    hypothesis, references = list(), list()

    for logits, text, summary in zip(logits, texts, summaries):
        scores = torch.sigmoid(logits)
        _, sort_idxs = torch.sort(scores, descending=True)

        cur_hypo = list()
        for idx in sort_idxs:
            sent_to_check = text[idx].strip()

            if trigram_block:
                if not has_same_trigram(sent_to_check, cur_hypo):
                    cur_hypo.append(sent_to_check)
            else:
                cur_hypo.append(sent_to_check)

            if len(cur_hypo) == extract_n_sents:
                break

        hypothesis.append("\n".join(cur_hypo))
        references.append("\n".join(summary))

    return hypothesis, references


def evaluate(model, dataloader, extract_nsents, device, pyrouge=True, trigram_block=True):
    model.eval()

    eval_loss = 0.0
    hypothesis, references = list(), list()
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            logits, _, _ = model(batch)

        loss = F.binary_cross_entropy(logits[0], batch.labels)

        eval_loss += loss.data

        nsents_per_item = torch.sum(batch.cls_mask[:, 0, :], dim=1).data.tolist()
        splited_logits = torch.split(logits[0], nsents_per_item, dim=0)
        hypos, refer = extract_oracle_from_logits(splited_logits, 
                                                  batch.texts,
                                                  batch.golden_tgt,
                                                  extract_nsents, 
                                                  trigram_block=trigram_block)

        hypothesis.extend(hypos)
        references.extend(refer)

    eval_loss /= len(dataloader)
    if pyrouge:
        rouge_scores = calc_rouge_from_pyrouge(hypothesis, references)
    else:
        rouge_scores = calc_rouge_from_python_implementation(hypothesis, references)

    r1, r2, rl = rouge_scores["rouge1_F1"], rouge_scores["rouge2_F1"], rouge_scores["rougel_F1"]
    log.info(f"Evaluate end | eval loss {eval_loss:5.4f} | rouge1_F1 {r1:5.2f} | rouge2_F1 {r2:5.2f} | rougel_F1 {rl:5.2f}")
    r1, r2, rl = rouge_scores["rouge1_R"], rouge_scores["rouge2_R"], rouge_scores["rougel_R"]
    log.info(f"Evaluate end | eval loss {eval_loss:5.4f} | rouge1_R {r1:5.2f} | rouge2_R {r2:5.2f} | rougel_R {rl:5.2f}")
    r1, r2, rl = rouge_scores["rouge1_P"], rouge_scores["rouge2_P"], rouge_scores["rougel_P"]
    log.info(f"Evaluate end | eval loss {eval_loss:5.4f} | rouge1_P {r1:5.2f} | rouge2_P {r2:5.2f} | rougel_P {rl:5.2f}")

    return eval_loss, rouge_scores

    