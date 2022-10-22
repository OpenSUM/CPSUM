import os
import json
import logging
import config
import os
from pathlib import Path
import torch
from pyarrow.json import read_json
from datetime import datetime


def get_current_datetime():
    return datetime.now().strftime('%Y_%m_%d_%H:%M:%S')


def get_logger(name):
    os.environ['PROJECT_ROOT'] = '/home/LAB/liujn/wym2/CPSUM'
    PROJECT_ROOT = Path(os.environ['PROJECT_ROOT'])
    NLI_LOGGING_DIR = PROJECT_ROOT / 'logging'

    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(filename=NLI_LOGGING_DIR / f'{get_current_datetime()}', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    #logging.basicConfig(filename=config.NLI_LOGGING_DIR / config.experiments_dir_name / config.experiments_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    logger = logging.getLogger(name)

    return logger


def read_jsonl(fp, l=-1):
    raw = read_json(fp)
    return WarpJsonObject(raw, l)


def write_jsonl(fp, data):
    with open(fp, "w") as f:
        f.writelines([json.dumps(it) + "\n" for it in data])


def append_jsonl(fp, data):
    with open(fp, "a") as f:
        f.writelines([json.dumps(it) + "\n" for it in data])


class WarpJsonObject:
    """convert pyarrow.Table to python dict"""
    def __init__(self, table, l):
        self._table = table
        self._feats = table.column_names
        self.l = l

        self._start_idx = 0

    def __len__(self):
        if self.l == -1:
            return len(self._table)
        else:
            return self.l

    def __getitem__(self, index: int):
        return {k: self._table[k][index].as_py() for k in self._feats}

    def __iter__(self):
        self._start_idx = -1
        return self

    def __next__(self):
        self._start_idx += 1

        if self._start_idx == len(self):
            raise StopIteration

        return self[self._start_idx]


def save_checkpoints(filename, ckpt):
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, "wb") as f:
        torch.save(ckpt, f)


def load_checkpoints(filename, device):
    obj = torch.load(filename, map_location=torch.device(device))
    return obj
