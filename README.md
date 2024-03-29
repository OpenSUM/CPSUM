# CPSUM

Source codes and data for COLING 2022 Paper "<a href="https://aclanthology.org/2022.coling-1.561">Noise-injected Consistency Training and Entropy-constrained Pseudo Labeling for Semi-supervised Extractive Summarization</a>"


## Data

You may first download complete dataset in Google Drive (URLs are in the table below) or <a href="https://pan.baidu.com/s/1rcRRUevdscAn9_TBWVBEtQ#list/path=%2F&parentPath=%2F">Baidu Netdisk</a>, and then overwrite the empty jsonl files in the "Data" folder.

|     | supervised train data | unsupervised train data | validation | test |
|  ----  | ----  | ---- | ---- | ---- |
| cnndm  | <a href="https://drive.google.com/drive/folders/1_iSISpr7Qgie3HheaHEB5mhRDjoOezQh?usp=sharing">train.jsonl</a> | <a href="">unsupervised_data.jsonl</a> | <a href="https://drive.google.com/drive/folders/1_iSISpr7Qgie3HheaHEB5mhRDjoOezQh?usp=sharing">val.jsonl</a> | <a href="https://drive.google.com/drive/folders/1_iSISpr7Qgie3HheaHEB5mhRDjoOezQh?usp=sharing">test.jsonl</a> |
| xsum  | <a href="https://drive.google.com/drive/folders/1RTvmPyVUZjQ93SD17BleyNp1TEMArQSk">train.jsonl</a> | <a href="">unsupervised_data.jsonl</a> | <a href="https://drive.google.com/drive/folders/1RTvmPyVUZjQ93SD17BleyNp1TEMArQSk">val.jsonl</a> | <a href="https://drive.google.com/drive/folders/1RTvmPyVUZjQ93SD17BleyNp1TEMArQSk">test.jsonl</a> |


You can manually select a specific amount of labeled data(10/100/1000) according to your needs (recommended), or without any pre-processing, the program will automatically select data at the top of data files.

Our unsupervised data are all from supervised datasets, but with label masking and data augmentation. You can use our processed data, or refer to <a href="https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT">tinybert</a> to augment the data yourself.

## Usage

The core program is ```train.py```. You can run the script in the "script" folder.

Take ```srcipt/cnndm.sh``` as an example:

```shell
#!/bin/bash


# prepare environment
source ### Fill in the path of conda.sh
conda activate ### Fill in the virtual environment name
export TOKENIZERS_PARALLELISM=false

# Seed
seed="42"

# Dataset
dataset_name="cnndm"
train_dataset_name="train_1000.jsonl"
unsupervised_dataset_name="unsupervised_train.jsonl"
supervised_filename="supervised_train_100_20.jsonl"


# Data
supervised_size="100"
ratio="4"
extract_nsents="3"


# Train
lr="2e-3"
weight_decay="1e-2"
lambdau="100"
num_warmup_steps="2000"
num_training_steps="5000"
loss_interval="5" 
val_interval="50"

# Ablation
PLform="hard"
do_consist="True"
do_pseudo_label="True"
do_rampup="True"  #
rampup_epoch="15" #

# Checkpoints
root_dir="experiments/cnndm/100/"
ckpt_dir="hard_cp_100_15"


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

```

All the above parameters can be modified as you like. The explanations of them can be viewed in ```arguments.py```

## Results

You may view the log files generated by the experiments in the "logging" folder.

## Citation

```
@inproceedings{wang-etal-2022-noise,
    title = "Noise-injected Consistency Training and Entropy-constrained Pseudo Labeling for Semi-supervised Extractive Summarization",
    author = "Wang, Yiming  and
      Mao, Qianren  and
      Liu, Junnan  and
      Jiang, Weifeng  and
      Zhu, Hongdong  and
      Li, Jianxin",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.561",
    pages = "6447--6456",
}
```
