from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
import fasttext
import pkg_resources
# import kenlm
import math

from transformers import BertModel, BertPreTrainedModel


class Evaluator(object):
    def __init__(self):
        resource_package = __name__

        # yelp_acc_path = 'acc_yelp.bin'
        # yelp_ppl_path = 'ppl_yelp.binary'
        yelp_ref0_path = 'yelp.refs.0'
        yelp_ref1_path = 'yelp.refs.1'

        # yelp_acc_file = pkg_resources.resource_stream(resource_package, yelp_acc_path)
        # yelp_ppl_file = pkg_resources.resource_stream(resource_package, yelp_ppl_path)
        yelp_ref0_file = pkg_resources.resource_stream(resource_package, yelp_ref0_path)
        yelp_ref1_file = pkg_resources.resource_stream(resource_package, yelp_ref1_path)

        self.yelp_ref = []
        with open(yelp_ref0_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())  # Positive refs
        with open(yelp_ref1_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())  # Negative refs
        # self.classifier_yelp = fasttext.load_model(yelp_acc_file.name)  # TODO: Need to replace this classifier
        # self.yelp_ppl_model = kenlm.Model(yelp_ppl_file.name)
        
    def yelp_style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(word_tokenize(text_transfered.lower().strip()))
        if text_transfered == '':
            return False
        # label = self.classifier_yelp.predict([text_transfered])  # TODO: Need to follow the replacement of `self.classifier_yelp`
        # style_transfered = label[0][0] == '__label__positive'
        style_transfered = 1
        return (style_transfered != style_origin)

    def yelp_acc_b(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.yelp_style_check(text, style):
                count += 1
        return count / len(texts)

    def yelp_acc_0(self, texts):
        styles_origin = [0] * len(texts)
        return self.yelp_acc_b(texts, styles_origin)

    def yelp_acc_1(self, texts):
        styles_origin = [1] * len(texts)
        return self.yelp_acc_b(texts, styles_origin)

    def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [word_tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = word_tokenize(text_transfered.lower().strip())
        return sentence_bleu(texts_origin, text_transfered) * 100

    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_0(self, texts_neg2pos):
        # assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[0], texts_neg2pos):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_1(self, texts_pos2neg):
        # assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[1], texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu(self, texts_neg2pos, texts_pos2neg):
        # assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        # assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 1000
        for x, y in zip(self.yelp_ref[0] + self.yelp_ref[1], texts_neg2pos + texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ppl(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            length += len(line.split())
            # score = self.yelp_ppl_model.score(line)  # TODO: need to replace the kenLM model with other library
            score = 0.0  # TODO: Replace with kenLM model
            sum += score
        return math.pow(10, -sum / length)


class StyleClassifier(BertPreTrainedModel):
    def __init__(self, args, clf_config):
        super(StyleClassifier, self).__init__(clf_config)
        self.args = args
        self.num_labels = args.num_labels
        self.clf_config = clf_config
        self.bert = BertModel.from_pretrained(args.bert_model)
        classifier_dropout = clf_config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(clf_config.hidden_size, args.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.args.problem_type is None:
                if self.num_labels == 1:
                    self.args.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.args.problem_type = "single_label_classification"
                else:
                    self.args.problem_type = "multi_label_classification"

            if self.args.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.args.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.args.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
