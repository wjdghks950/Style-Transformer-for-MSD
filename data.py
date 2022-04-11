import time, logging, os, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.legacy import data
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer

from utils import tensor2text

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MSDExample(object):
    def __init__(self, text, label, concepts):
        '''
        text - text for the MSD text sequence (e.g., expert or laymen sentence)
        label - 0 or 1 (0: expert / 1: laymen)
        concepts - List[Dict] - (e.g., [{"range": [2,3], "term": "dyspnea", "cui":["C1963100", "C001333"]}])
        '''
        self.text = text
        self.label = label
        self.concepts = concepts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        if self.text:
            s += "TEXT: {}".format(self.text)
        if self.label:
            s += "\nLABEL: {}".format(self.label)
        if self.concepts:
            s += "\nTERM: {} (range - {})".format(self.concepts[0]['term'], self.concepts[0]['range'])
            s += "\nCUI: ({})".format(self.concepts[0]['cui'])
        return s


class MSDFeature(object):
    def __init__(self, input_ids, input_mask, label, max_seq_len, concepts, tokens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label
        self.max_seq_len = max_seq_len
        self.concepts = concepts
        self.tokens = tokens  # The tokenized input sequence
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "[Feature sample]\n"
        if self.tokens and self.input_ids:
            s += "TEXT: {}".format(self.tokens)
        if self.input_ids:
            s += "\nInput IDs: {}".format(self.input_ids)
        if self.label:
            s += "\nLABEL: {}".format(self.label)
        if self.concepts:
            s += "\nTERM: {} (range - {})".format(self.concepts[0]['term'], self.concepts[0]['range'])
            s += "\nCUI: ({})\n".format(self.concepts[0]['cui'])
        return s


class MSDDataset(Dataset):
    def __init__(self, path, data_dir, config, mode="train"):
        self.config = config
        self.mode = mode
        try:
            self.examples = self._read_examples(os.path.join(data_dir, path))
        except Exception as e:
            print(e + "{} does not exist".format(os.path.join(data_dir, path)))

        if config.train_styleclf:  # Activate when training the bert-based StyleClassifier
            model_prefix = config.bert_model.split("-")[0].strip()
            if model_prefix == "bert":
                tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
            elif model_prefix == "roberta":
                tokenizer = RobertaTokenizer.from_pretrained(config.bert_model)
            elif model_prefix == "albert":
                tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge')
            else:
                raise AttributeError("Specified attribute {} is not found".format(config.bert_model))

            # Construct features out of examples (MSDExamples -> MSDFeatures)
            features = self._create_features_from_examples(self.examples, tokenizer, mode=mode)

            self.input_ids = torch.tensor([f.input_ids for f in features]).long()
            self.input_masks = torch.tensor([f.input_mask for f in features]).long()
            self.labels = torch.tensor([f.label for f in features]).long()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return a single MSDFeature upon function call
        return (self.input_ids[idx], self.input_masks[idx], self.labels[idx])

    def _read_examples(self, path):
        examples = []
        if os.path.exists(path):
            with open(path, mode="r") as fp:
                txt_dict = [d for d in fp.readlines()]
            # Reconstruct data as a list of dicts
            data_dict = [json.loads(dict_inst) for dict_inst in txt_dict]
            for i, msd_inst in enumerate(data_dict):
                examples.append(MSDExample(msd_inst['text'], msd_inst['label'], msd_inst['concepts']))
            assert len(data_dict) == len(examples)
        else:
            raise FileNotFoundError("Path does not exist")
        return examples

    def _create_features_from_examples(self, examples, tokenizer, mode="train"):
        '''
        examples - list of MSDExample
        tokenizer - One of `transformers` pre-trained tokenizers (e.g., BertTokenizer)
        '''
        logger.info('Creating features from `examples (MSDExample -> MSDFeatures)`')
        logger.info('Using [ {} ] tokenizer'.format(self.config.bert_model))

        max_seq_len = self.config.max_seq_len
        enable_toks = True
        features = []
        for (example_idx, example) in tqdm(enumerate(examples), total=len(examples), desc="[{}] Convert MSDExample to MSDFeature".format(mode)):
            tokens = tokenizer.tokenize(example.text)
            tokenizer_out = tokenizer(example.text)
            input_ids = tokenizer_out['input_ids']
            input_mask = tokenizer_out['attention_mask']
            token_type_ids = tokenizer_out['token_type_ids']

            # pad by 0 to fill the `max_seq_len`
            input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
            input_mask = input_mask + [0] * (max_seq_len - len(input_mask))
            token_type_ids = token_type_ids + [0] * (max_seq_len - len(token_type_ids))

            assert len(input_ids) == len(input_mask) == len(token_type_ids)

            # find a valid span for the concept words
            valid_concepts = []
            for concept in example.concepts:
                val_concept = self.find_valid_span(tokens, concept, tokenizer)
                valid_concepts.append(val_concept)
            assert len(valid_concepts) == len(example.concepts)
            concepts = valid_concepts

            feature = MSDFeature(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        label=example.label,
                        max_seq_len=max_seq_len,
                        concepts=concepts,
                        tokens=tokens if enable_toks else None
                        )
            features.append(feature)

            if example_idx % 10000 == 0:
                logger.info(feature)
        
        assert len(features) == len(examples)
        return features

    @staticmethod
    def find_valid_span(tokens, concept_dict, tokenizer):
        '''
        Find a valid span within the tokenized text (i.e., `tokens`) for a given `concept_dict`.
        '''
        concept_term = tokenizer.tokenize(concept_dict["term"])
        tok_offset = len(concept_term)
        for tok_idx in range(0, len(tokens)):
            if tokens[tok_idx : tok_idx + tok_offset] == concept_term:  # Exact match
                return {"range": [tok_idx, tok_idx + tok_offset],
                        "term": concept_dict["term"],
                        "tokenized_term": concept_term,
                        "cui": concept_dict["cui"]}

        # In case of different word form (e.g., plural), check if the concept term is a substring
        reconst_tokens = tokenizer.convert_tokens_to_string(tokens)
        if concept_dict["term"] in reconst_tokens:
            for tok_idx in range(0, len(tokens)):
                if tokens[tok_idx][:len(concept_term[0])].strip() == concept_term[0].strip():
                    # First word of the concept word(s) match, then return the new `concept_dict`
                    return {"range": [tok_idx, tok_idx + tok_offset],
                            "term": concept_dict["term"],
                            "tokenized_term": concept_term,
                            "cui": concept_dict["cui"]}

            raise AttributeError("Concept word [{}] [tokenized:{}] not found within text\n[TEXT:\"{}\"].".format(concept_dict["term"], concept_term, tokens))


class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text


class MSDIterator(object):
    def __init__(self, exp_iter, lay_iter):
        self.exp_iter = exp_iter
        self.lay_iter = lay_iter

    def __iter__(self):
        for batch_exp, batch_lay in zip(iter(self.exp_iter), iter(self.lay_iter)):
            if batch_exp.text.size(0) == batch_lay.text.size(0):
                yield batch_exp.text, batch_lay.text


def convert_msd_to_datafield(config, train_path, test_path):
    train_dataset = MSDDataset(path=train_path, data_dir=config.data_path, config=config, mode="train")
    eval_dataset = MSDDataset(path=test_path, data_dir=config.data_path, config=config, mode="eval")

    # Create an `expert` to `laymen` data files to be used by `data.TabularDataset`
    export_msd_to_type(config, train_dataset, eval_dataset)


def export_msd_to_type(config, train_dataset, eval_dataset):
    root = config.data_path
    expert_train = "expert_train.txt"
    laymen_train = "laymen_train.txt"
    expert_test = "expert_test.txt"  # Unlike `train` dataset, the `test` dataset is a parallel corpus
    laymen_test = "laymen_test.txt"
    if any([not os.path.exists(os.path.join(root, path)) for path in [expert_train, laymen_train, expert_test, laymen_test]]):
        with open(os.path.join(root, expert_train), 'w') as exp_f,\
             open(os.path.join(root, laymen_train), 'w') as lay_f:
            for example in tqdm(train_dataset.examples, desc="Reading [MSD Train Dataset]\n"):
                if example.label == 0:
                    exp_f.write(f"{example.text}\n")
                else:
                    lay_f.write(f"{example.text}\n")
        with open(os.path.join(root, expert_test), 'w') as exp_f_test,\
             open(os.path.join(root, laymen_test), 'w') as lay_f_test:
            for example in tqdm(eval_dataset.examples, desc="Reading [MSD Test Dataset]\n"):
                if example.label == 0:
                    exp_f_test.write(f"{example.text}\n")
                else:
                    lay_f_test.write(f"{example.text}\n")
        assert os.path.exists(os.path.join(root, expert_train))
        assert os.path.exists(os.path.join(root, laymen_train))
        assert os.path.exists(os.path.join(root, expert_test))
        assert os.path.exists(os.path.join(root, laymen_test))
        logger.info("Data file export complete for [expert] and [laymen]")
    else:
        logger.info("[ Data file already exists for [expert] and [laymen] ]")


def load_msd_dataset(config, train_path='train.txt', test_path='test.txt'):
    '''
    Returns train and test dataset iterators for the MSD dataset along with the `vocab`.
    '''
    root = config.data_path
    convert_msd_to_datafield(config, train_path, test_path)
    TEXT = data.Field(batch_first=True, fix_length=config.max_length, eos_token='<eos>')

    dataset_fn = lambda name: data.TabularDataset(
        path=os.path.join(root, name),
        format='tsv',
        fields=[('text', TEXT)]
    )
    expert_train = "expert_train.txt"
    laymen_train = "laymen_train.txt"
    expert_test = "expert_test.txt"  # Unlike `train` dataset, the `test` dataset is a parallel corpus
    laymen_test = "laymen_test.txt"
    train_exp_set, train_lay_set = map(dataset_fn, [expert_train, laymen_train])
    test_exp_set, test_lay_set = map(dataset_fn, [expert_test, laymen_test])

    # Construct a Vocab object from one or more datasets
    TEXT.build_vocab(train_exp_set, train_lay_set, min_freq=config.min_freq)
    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )
    train_exp_iter, train_lay_iter = map(lambda x: dataiter_fn(x, True), [train_exp_set, train_lay_set])
    test_exp_iter, test_lay_iter = map(lambda x: dataiter_fn(x, False), [test_exp_set, test_lay_set])

    train_iters = MSDIterator(train_exp_iter, train_lay_iter)
    test_iters = MSDIterator(test_exp_iter, test_lay_iter)

    return train_iters, test_iters, vocab


def load_dataset(config, train_pos='train.pos', train_neg='train.neg',
                 dev_pos='dev.pos', dev_neg='dev.neg',
                 test_pos='test.pos', test_neg='test.neg'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors = torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_pos_iter, train_neg_iter = map(lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set])
    dev_pos_iter, dev_neg_iter = map(lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set])
    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)

    return train_iters, dev_iters, test_iters, vocab


if __name__ == '__main__':
    class Config():
        data_path = './data/yelp/'
        min_freq = 3
        batch_size = 1
        load_pretrained_embed = False
        embed_size = 256
        device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')

    config = Config()
    train_iter, _, _, vocab = load_dataset(config)
    print("[ Vocab size: {} ]".format(len(vocab)))
    for batch in train_iter:
        text_pos = tensor2text(vocab, batch[0])
        text_neg = tensor2text(vocab, batch[1])
        print('\n[Original Text (pos)]\n{}\n'.format(text_pos))
        print('[Original Text (neg)]\n{}\n'.format(text_neg))
        break
