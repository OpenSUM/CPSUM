#!/bin/bash


# prepare environment
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate textsum

export TOKENIZERS_PARALLELISM=false

# Seed
seed="42"

# Dataset
dataset_name="xsum"
train_dataset_name="train.jsonl"
unsupervised_dataset_name="unsupervised_train.jsonl"
supervised_filename="supervised_train.jsonl"


# Data
supervised_size="100"  #
ratio="4"
extract_nsents="1"


# Train
lr="2e-3"
weight_decay="1e-2"
lambdau="100"
num_warmup_steps="2000" #
num_training_steps="5000"  #
loss_interval="5" #
val_interval="50"  #

# Ablation
PLform="hard" #
do_consist="True" #
do_pseudo_label="True"  #
do_rampup="True"  #
rampup_epoch="10" #

# Checkpoints
root_dir="experiments/xsum/10/"
ckpt_dir="hard_cp_10"


python src/train.py --seed $seed \
                    --dataset_name $dataset_name \
                    --train_dataset_name $train_dataset_name \
                    --unsupervised_dataset_name $unsupervised_dataset_name \
                    --supervised_filename $supervised_filename \
                    --supervised_size $supervised_size \
                    --ratio $ratio \
                    --extract_nsents $extract_nsents \
                    --lr $lr \
                    --weight_decay $weight_decay \
                    --lambdau $lambdau \
                    --num_warmup_steps $num_warmup_steps \
                    --num_training_steps $num_training_steps \
                    --loss_interval $loss_interval \
                    --val_interval $val_interval \
                    --PLform $PLform \
                    --do_consist $do_consist \
                    --do_pseudo_label $do_pseudo_label \
                    --do_rampup $do_rampup \
                    --rampup_epoch $rampup_epoch \
                    --root_dir $root_dir \
                    --ckpt_dir $ckpt_dir \
