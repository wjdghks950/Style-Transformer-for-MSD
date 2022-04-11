import torch
import neptune
import time, logging, argparse
from data import load_msd_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
from transformers import BartModel

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Config():
    data_path = './data/MSD_dataset'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Cond'  # 'Multi' or 'Cond'
    task = "msd"  # Tasks \in ["msd", "sentiment"]

    load_pretrained_embed = False
    train_styleclf = False
    min_freq = 3
    max_length = 128
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2  # Styles \in {Expert, Laymen}
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 8
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 500
    eval_steps = 1000
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0


def main():
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/MSD_dataset", type=str, help="Dataset path for MSD dataset.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='./model_ckpt',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_inference",
                        action='store_true',
                        help="Whether to run inference on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--logging_steps", type=int, default=1000, help="Logging steps during the training iterations.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_false',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--init_weights_dir",
                        default='',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # # TODO: Implement the model part - Which model to use? Can BERT-based models (e.g., ALBERT) be used in a generative framework?
    # # TODO: How about BART?
    # if args.init_weights_dir:
    #     model = BertTransformer.from_pretrained(args.init_weights_dir)
    # else:
    #     model = BertTransformer.from_pretrained(args.bert_model)

    # # Set model device to cuda (or cpu if there is none)
    # model.to(device)
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    train_iters, test_iters, vocab = load_msd_dataset(config)
    print("[ Vocab size: {} ]".format(len(vocab)))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)

    if args.logging_steps > 0:
        neptune.init(
            project_qualified_name="wjdghks950/StyleTransfer",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NzQ2OTM5ZC02N2Y5LTQ2NDEtYWU3NS0yNWY3YWI1NDgzMjEifQ==")
        neptune.create_experiment(name=" ({}) - MSD Style Transformer".format(args.bert_model))
        neptune.append_tag("StyleTransformer", "Expert-Laymen")

    # train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    train(config, vocab, model_F, model_D, train_iters, test_iters, test_iters)
    

if __name__ == '__main__':
    main()
