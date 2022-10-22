import os
import torch
import utils
from pathlib import Path

os.environ['PROJECT_ROOT'] = '/home/LAB/liujn/wym2/ConsistSum'
PROJECT_ROOT = Path(os.environ['PROJECT_ROOT'])

# Logging
NLI_LOGGING_DIR = PROJECT_ROOT / 'logging'

seed = 42                                 # Random seed.


# GPU
gpu = 0                                       # GPU ID to use.
cuda = 1                                     # GPU or CPU.

# Dataset
train_batch_size = 4                          # Batch size of training.
test_batch_size = 32                          # Batch size of validation.
val_batch_size = 32                           # Batch size of testing.
src_max_len = 512
max_text_ntokens_per_sent = 200
min_text_ntokens_per_sent = 5
num_workers = 8                               # Number of process workers in dataloader.
supervised_size = 1000                       # Number of supervised data in consistency training.
unsupervised_size = 4000                     # Number of unsupervised data in consistency training.
extract_nsents = 1                            # Number of oracle summary.

# Data augmentation
augmented_num = 4
glove_filename = 'glove.42B.300d.txt'

# Training
lr = 2e-3                                     # Base learning rate.
weight_decay = 1e-2                           # Weight decay of adamW.
num_warmup_steps = 2000                      # Total number of warmup.
num_training_steps = 20000                    # Total number of training steps.
loss_interval = 5
val_interval = 500                           # Total number of warmup.
lambdau = 100                                 # Hyperparameters used in the sharpen process.
do_block = True                               # Trigram block or not.

# Test 
test_after_train = True                       # Do test after training.

# Checkpoints
root_dir = "experiments/cnndm/consist/"       # The root directory of this run.
ckpt_dir = "checkpoints"                      # The specific directory name in the root directory to save checkpoints.
resume_ckpt_path = None                       # Resume checkpoint path.

# Pre-trained model
encoder_name_or_path = "bert-base-uncased"    # The name or path of pretrained language model.
tokenizer_name_or_path = "bert-base-uncased"  # The name or path of pretrained tokenizer.
config_path = None                            # The path of the config file.



