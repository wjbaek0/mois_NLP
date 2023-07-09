import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModelWithLMHead
# from pytorch_pretrained_bert import BertModel, BertConfig
from torchcrf import CRF

import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
# 'hidden_size': 768,
bert_config = {'attention_probs_dropout_prob': 0.1,
               'hidden_act': 'gelu',
               'hidden_dropout_prob': 0.1,
               'hidden_size': 768,
               'initializer_range': 0.02,
               'intermediate_size': 3072,
               'max_position_embeddings': 512,
               'num_attention_heads': 12,
               'num_hidden_layers': 12,
               'type_vocab_size': 2,
               'vocab_size': 8002}

kcbert_config = {
    "max_position_embeddings": 300,
    "hidden_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30000,
    "hidden_size": 768,
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "architectures": [
        "BertForMaskedLM"
    ],
    "model_type": "bert"
}
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bert = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

class KobertBiLSTMCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, time_distribute=False, vocab=None) -> None:
        super(KobertBiLSTMCRF, self).__init__()
        if vocab is None:  # pretraining model 사용
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
            self.bert = BertModel.from_pretrained(
                'skt/kobert-base-v1', return_dict=False)
            self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(
                tokenizer.vocab_file, padding_token='[PAD]')
        else:  # finetuning model 사용
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]
        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) //
                              2, dropout=config.dropout, batch_first=True, bidirectional=True)
        if time_distribute == True:
            self.position_wise_ff = TimeDistributed(
                nn.Linear(config.hidden_size, num_classes), batch_first=True)
        else:
            self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=False):
        seq_length = input_ids.ne(self._pad_id).sum(dim=1).cpu()
        attention_mask = input_ids.ne(self._pad_id).float()
        outputs = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            pack_padded_last_encoder_layer = pack_padded_sequence(
                last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, hc = self.bilstm(pack_padded_last_encoder_layer)
            outputs = pad_packed_sequence(
                outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ff(outputs)

        if tags is not None:  # crf training
            log_likelihood, sequence_of_tags = self.crf(
                emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else:  # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags


class KcbertBiLSTMCRF(nn.Module):
    """ kcBERT with CRF """

    def __init__(self, config, num_classes, time_distribute=False, vocab=None) -> None:
        super(KcbertBiLSTMCRF, self).__init__()
        if vocab is None:  # pretraining model 사용
            tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
            self.bert = AutoModelWithLMHead.from_pretrained(
                "beomi/kcbert-base")
            self.vocab = AutoTokenizer.from_pretrained("beomi/kcbert-base")
            self.bert.cls.predictions = self.bert.cls.predictions.transform
            self.bert.cls.predictions.LayerNorm = torch.nn.Tanh()
        else:  # finetuning model 사용
            self.bert = BertModel(config=BertConfig.from_dict(kcbert_config))
            self.vocab = vocab
        self._pad_id = self.vocab.vocab[self.vocab.pad_token]
        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) //
                              2, dropout=config.dropout, batch_first=True, bidirectional=True)
        if time_distribute == True:
            self.position_wise_ff = TimeDistributed(
                nn.Linear(config.hidden_size, num_classes), batch_first=True)
        else:
            self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=False):
        seq_length = input_ids.ne(self._pad_id).sum(dim=1).cpu()
        attention_mask = input_ids.ne(self._pad_id).float()
        outputs = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            pack_padded_last_encoder_layer = pack_padded_sequence(
                last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, hc = self.bilstm(pack_padded_last_encoder_layer)
            outputs = pad_packed_sequence(
                outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ff(outputs)

        if tags is not None:  # crf training
            log_likelihood, sequence_of_tags = self.crf(
                emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else:  # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))
        return y
