from cgi import print_form
import os
import math
import traceback
import random
import logging
import config
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.logging import set_verbosity_error

from arguments import parse_args
from evaluate import evaluate
from model import CPSUM
from data import CPSUMDataModule
from utils import (
    load_checkpoints,
    save_checkpoints,
    get_logger,
    write_jsonl
)

log = get_logger(__name__)


def calc_ent(x):
    ent = 0.0
    for value in x:
        if value > 0:
            ent -= value * math.log(value, 2)
    return ent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_rampup(current, rampup_length):
    current = np.clip(current / rampup_length, 0.0, 1.0)
    return float(current)


def get_optimizer_and_scheduler(model):
    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    is_decay_params = lambda n, p: n in decay_parameters and p.requires_grad
    is_nodecay_params = lambda n, p: n not in decay_parameters and p.requires_grad

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if is_decay_params(n, p)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if is_nodecay_params(n, p)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)

    def lr_lambda(current_step):
        current_step = current_step + 1
        return min(current_step ** -0.5, current_step * (config.num_warmup_steps ** -1.5))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler


def cpsum_train(args):
    model = CPSUM(args.encoder_name_or_path, args)
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model)

    datamodule = CPSUMDataModule(dataset_name=args.dataset_name,
                                 train_batch_size=args.train_batch_size,
                                 val_batch_size=args.val_batch_size,
                                 test_batch_size=args.test_batch_size,
                                 tokenizer_name_or_path=args.tokenizer_name_or_path,
                                 train_dataset_name=args.train_dataset_name,
                                 val_dataset_name=args.val_dataset_name,
                                 test_dataset_name=args.test_dataset_name,
                                 unsupervised_dataset_name=args.unsupervised_dataset_name,
                                 num_workers=args.num_workers,
                                 src_max_len=args.src_max_len,
                                 max_text_ntokens_per_sent=args.max_text_ntokens_per_sent,
                                 min_text_ntokens_per_sent=args.min_text_ntokens_per_sent,
                                 )
    datamodule.prepare(do_consist_training=True, supervised_dataset_size=args.supervised_size,
                       supervised_filename=args.supervised_filename)

    train_dataloader, unsupervised_dataloader = datamodule.train_dataloader(), datamodule.unsupervised_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = iter(train_dataloader)
    unsupervised_dataiter = iter(unsupervised_dataloader)

    device = torch.device("cpu")
    if args.cuda:
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        model = model.to(torch.device(args.gpu))

    current_step = 0
    train_loss = 0.0
    supervised_loss = 0.0
    unsupervised_loss = 0.0
    raw_unsupervised_loss = 0.0
    best_eval_loss = np.inf
    best_rouge_score = np.NINF
    best_loss_checkpoints_filename = None
    best_rouge_checkpoints_filename = None
    if config.resume_ckpt_path is not None:
        log.info(f"Resume from {args.resume_ckpt_path}...")
        ckpt = load_checkpoints(args.resume_ckpt_path, args.gpu if args.cuda else "cpu")
        current_step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])

    pseudo_data = []
    val = []
    flag = 0
    log.info("Start training!")
    while current_step < args.num_training_steps:
        model.train()

        try:
            batch = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        batch = batch.to(device)
        logits, pseudo_labels, _ = model(batch)
        supervised_ent = sum([calc_ent(i) for i in pseudo_labels])
        supervised_ent /= args.train_batch_size

        bce_loss = F.binary_cross_entropy(logits[0], batch.labels)

        if args.do_consist:
            consist_loss = 0
            for i in range(args.ratio):
                try:
                    unsupervised_batch = next(unsupervised_dataiter)
                except StopIteration:
                    unsupervised_dataiter = iter(unsupervised_dataloader)
                    unsupervised_batch = next(unsupervised_dataiter)
                except Exception as e:
                    log.error(f"Error when loading data: {e}")
                    log.error(traceback.format_exc())
                    exit()

                unsupervised_batch = unsupervised_batch.to(device)
                unsupervised_logits, unsupervised_pseudo_labels, hardPLs = model(unsupervised_batch)

                for j in range(args.train_batch_size):
                    unsupervised_ent = calc_ent(unsupervised_pseudo_labels[j])
                    prop = random.random()
                    if unsupervised_ent / len(unsupervised_pseudo_labels[j]) < supervised_ent / (sum([len(i) for i in pseudo_labels]) / len(pseudo_labels)):
                        if args.do_rampup == 0:
                            pseudo_data.append({"src": unsupervised_batch.texts[j],
                                                "labels": unsupervised_pseudo_labels[j],
                                                "golden_tgt": unsupervised_batch.golden_tgt[j]})
                        else:
                            if prop < min(current_step / (args.supervised_size // args.train_batch_size) / args.rampup_epoch, 1):
                                pseudo_data.append(
                                    {"src": unsupervised_batch.texts[j],
                                     "labels": (hardPLs[j] if args.PLform == 'hard' else unsupervised_pseudo_labels[j]),
                                     "golden_tgt": unsupervised_batch.golden_tgt[j]})

                consist_loss += F.mse_loss(unsupervised_logits[0], unsupervised_logits[1])

            weighted_consist_loss = config.lambdau * linear_rampup(current_step, 10000) * consist_loss
            loss = bce_loss + weighted_consist_loss
        
        else:
            loss = bce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        current_step = current_step + 1
        train_loss += loss.data.item()
        supervised_loss += bce_loss.data.item()

        if args.do_consist:
            raw_unsupervised_loss += consist_loss.data.item()
            unsupervised_loss += weighted_consist_loss.data.item()

        if args.do_pseudo_label and flag == 0:
            if len(val) > 3:
                if val[-1] <= val[-2] <= val[-3] <= val[-4]:
                    datamodule.append_pseudo(pseudo_data, args.supervised_filename)
                    train_dataloader = datamodule.train_dataloader()
                    train_dataiter = iter(train_dataloader)
                    flag = 1

        if current_step % config.loss_interval == 0:
            if args.do_consist:
                log.info(
                    f"Step {current_step:3d} | train loss {(train_loss / config.loss_interval):5.4f} "
                    f"| supervised loss {(supervised_loss / config.loss_interval):5.4f} "
                    f"| raw_unsupervised loss {(raw_unsupervised_loss / config.loss_interval):5.4f} "
                    f"| unsupervised loss {(unsupervised_loss / config.loss_interval):5.4f}")
                supervised_loss = 0.0
                unsupervised_loss = 0.0
                raw_unsupervised_loss = 0.0

            else:
                log.info(f"Step {current_step:3d} | train loss {(train_loss / 100.0):5.4f}")

            train_loss = 0.0

        if current_step % args.val_interval == 0:
            eval_loss, rouge_scores = evaluate(model,
                                               val_dataloader,
                                               config.extract_nsents,
                                               device,
                                               pyrouge=False,
                                               trigram_block=args.do_block)
            val.append(rouge_scores["rouge1_F1"])

            checkpoints = {
                "step": current_step,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
            }
            checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
            save_checkpoints(checkpoints_filename, checkpoints)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_loss_checkpoints_filename = checkpoints_filename
            tmp_rouge_score = rouge_scores["rouge1_F1"] + rouge_scores["rouge2_F1"] + rouge_scores["rougel_F1"]
            if tmp_rouge_score > best_rouge_score:
                best_rouge_score = tmp_rouge_score
                best_rouge_checkpoints_filename = checkpoints_filename

    log.info("Train end.")
    log.info(f"The best loss checkpoint file is in {best_loss_checkpoints_filename}")
    log.info(f"The best rouge checkpoint file is in {best_rouge_checkpoints_filename}")

    if args.test_after_train:
        ckpt = load_checkpoints(best_loss_checkpoints_filename, args.gpu if args.cuda else "cpu")
        model.load_state_dict(ckpt["model"])
        log.info("Test the best loss checkpoints.")
        evaluate(model,
                 test_dataloader,
                 args.extract_nsents,
                 device,
                 pyrouge=True,
                 trigram_block=args.do_block)
        ckpt = load_checkpoints(best_rouge_checkpoints_filename, args.gpu if args.cuda else "cpu")
        model.load_state_dict(ckpt["model"])
        log.info("Test the best rouge checkpoints.")
        evaluate(model,
                 test_dataloader,
                 args.extract_nsents,
                 device,
                 pyrouge=True,
                 trigram_block=args.do_block)

    return


if __name__ == "__main__":
    set_verbosity_error()
    args = parse_args()
    log.info(f"dataset: {args.dataset_name} | "
             f"supervised data size: {args.supervised_size} | "
             f"do consistency training: {args.do_consist} | "
             f"do pseudo labeling: {args.do_pseudo_label} | "
             f"do ramp-up exploitation: {args.do_rampup} | "
             f"ramp-up epoch: {args.rampup_epoch}")

    if config.seed > 0:
        log.info(f"Set seed to {args.seed}")
        seed_everything(args.seed)
    else:
        log.info(f"Set random seed")

    cpsum_train(args)

