'''
Script to train the `StyleClassifier` to measure the style accuracy of the style-transferred sentences.
'''
import logging
import os
import torch
import numpy as np
import neptune
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from evaluator import StyleClassifier
from data import MSDDataset
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import compute_metrics


# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = BertConfig.from_pretrained(args.bert_model)
        self.model = StyleClassifier(args, clf_config=config)
        # `self.train_dataset` and `self.test_dataset` are
        # padded and in the following input sequence form: [cls] text [sep]
        self.train_dataset = MSDDataset("train.txt", args.data_dir, args, mode="train")
        self.test_dataset = MSDDataset("test.txt", args.data_dir, args, mode="test")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=train_sampler,
                                      batch_size=self.args.train_batch_size,
                                      num_workers=2)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train logger info
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)

        global_step = 0
        tr_loss, tr_ppl = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                loss = loss / self.args.gradient_accumulation_steps  # normalize loss to account for batch accumulation

                tr_loss += loss.item()
                tr_ppl += torch.exp(loss)  # Calculate perplexity for every step

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    neptune.log_metric('(Train) Loss', tr_loss)
                    neptune.log_metric('(Train) Perplexity', tr_ppl)

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logger.info('(Train) Loss: %f', tr_loss)
                        logger.info('(Train) Perplexity: %f', tr_ppl)
                        self.evaluate("test")
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        else:
            raise Exception("Only test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2]}

                outputs = self.model(**inputs)  # (loss), logits, (hidden_states), (attentions)
                tmp_eval_loss, logits = outputs[0], outputs[1]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        # if self.early_stopping.validate((results['loss'])):
        #     print("Early stopping... Terminating Process.")
        #     exit(0)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            neptune.log_metric("(Eval.) {}".format(key), results[key])
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_ckpt", type=str, help="Save path for model checkpoint.")
    parser.add_argument("--data_dir", default="data/MSD_dataset", type=str, help="Dataset path for MSD dataset.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Flag for lower case or not in the tokenizer.")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="Maximum sequence length for the model input sequence.")
    parser.add_argument("--output_dir",
                        default='./model_ckpt',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--problem_type", default="single_label_classification", type=str, help="Type of classification for the given task.")
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
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--logging_steps", type=int, default=5000, help="Logging steps during the training iterations.")
    parser.add_argument("--num_labels", default=2, type=int, help="Number of labels (# of classes).")                        
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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

    trainer = Trainer(args)
    if args.logging_steps > 0:
        neptune.init(
            project_qualified_name="wjdghks950/StyleTransfer",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NzQ2OTM5ZC02N2Y5LTQ2NDEtYWU3NS0yNWY3YWI1NDgzMjEifQ==")
        neptune.create_experiment(name=" ({}) - Style Classifier".format(args.bert_model))
        neptune.append_tag("BertForSequenceClassification", "StyleClassifier", "Expert-Laymen")
    logger.info(f"Training Style Classifier using [{args.bert_model}]")
    trainer.train()
    neptune.stop()
