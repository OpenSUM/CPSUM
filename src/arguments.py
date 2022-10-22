from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    subparser = parser.add_argument_group("Seed")
    subparser.add_argument("--seed", type=int, default=42, help="seed")

    subparser = parser.add_argument_group("GPU")
    subparser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    subparser.add_argument("--cuda", type=int, default=1, help="GPU or CPU.")

    subparser = parser.add_argument_group("Dataset")
    subparser.add_argument("--dataset_name", type=str, required=True, default="cnndm", help="news dataset.")
    subparser.add_argument("--train_dataset_name", type=str, default="train.jsonl", help="filename of train dataset")
    subparser.add_argument("--val_dataset_name", type=str, default="val.jsonl", help="filename of validation dataset")
    subparser.add_argument("--test_dataset_name", type=str, default="test.jsonl", help="filename of test dataset")
    subparser.add_argument("--unsupervised_dataset_name", type=str, default="unsupervised_train.jsonl",
                           help="filename of unsupervised train dataset")
    subparser.add_argument("--supervised_filename", type=str, help="filename of supervised data(raw + pseudo)")

    subparser = parser.add_argument_group("Data")
    subparser.add_argument("--train_batch_size", type=int, default=4, help="Batch size of training.")
    subparser.add_argument("--val_batch_size", type=int, default=32, help="Batch size of validation.")
    subparser.add_argument("--test_batch_size", type=int, default=32, help="Batch size of testing.")
    subparser.add_argument("--supervised_size", type=int, default=100,
                           help="Number of supervised data in consistency training.")
    subparser.add_argument("--ratio", type=int, default=4,
                           help="the ratio of unlabeled and labeled data in each training step")
    subparser.add_argument("--num_workers", type=int, default=8, help="number of process workers in dataloader")
    subparser.add_argument("--src_max_len", type=int, default=512, help="maximum length of input text")
    subparser.add_argument("--max_text_ntokens_per_sent", type=int, default=200,
                           help="maximum number of tokens per sentence")
    subparser.add_argument("--min_text_ntokens_per_sent", type=int, default=5,
                           help="minimum number of tokens per sentence")
    subparser.add_argument("--extract_nsents", type=int, default=3, help="number of oracle summary")

    subparser = parser.add_argument_group("Pre-trained model")
    subparser.add_argument("--encoder_name_or_path", type=str, default="bert-base-uncased", help="encoder_name_or_path")
    subparser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased",
                           help="tokenizer_name_or_path")

    subparser = parser.add_argument_group("Train")
    subparser.add_argument("--lr", type=float, help="base learning rate")
    subparser.add_argument("--weight_decay", type=float, help="weight decay of adamW")
    subparser.add_argument("--hidden_size", type=int, default=768, help="hidden_size")
    subparser.add_argument("--lambdau", type=int, default=100, help="hyper-parameters used in the sharpen process")
    subparser.add_argument("--do_block", type=bool, default=True, help="trigram block or not")
    subparser.add_argument("--num_warmup_steps", type=int, default=2000, help="Total number of warmup steps.")
    subparser.add_argument("--num_training_steps", type=int, default=10000, help="Total number of training steps.")
    subparser.add_argument("--loss_interval", type=int, default=5, help="loss_interval")
    subparser.add_argument("--val_interval", type=int, default=100, help="val_interval")

    subparser = parser.add_argument_group("Test")
    subparser.add_argument("--test_after_train", type=bool, default=True, help="do test after training")

    subparser = parser.add_argument_group("Ablation")
    subparser.add_argument("--PLform", type=str, default="soft", choices=['soft', 'hard'],
                           help="the type of pesudo labels")
    subparser.add_argument("--do_consist", type=bool, default=False, help="Consistency training or not.")
    subparser.add_argument("--do_pseudo_label", type=bool, default=False, help="Pseudo-labeling or not.")
    subparser.add_argument("--do_rampup", type=bool, default=True, help="Rampup exploitation or not.")
    subparser.add_argument("--rampup_epoch", type=int, default=10, help="rampup epoch")

    subparser = parser.add_argument_group("Checkpoints")
    subparser.add_argument("--root_dir", type=str, default="./experiments/cnndm",
                           help="The root directory of this run.")
    subparser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                           help="The specific directory name in the root directory to save checkpoints.")
    subparser.add_argument("--resume_ckpt_path", type=bool, default=False, help="resume checkpoint path")

    args = parser.parse_args()
    return args
