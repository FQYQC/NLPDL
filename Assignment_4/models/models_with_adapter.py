from transformers.models.bert.modeling_bert import (
    BertModel,
    BertSelfOutput,
    BertEncoder,
    BertOutput,
    BertLayer,
    BertSelfAttention,
    BertAttention
)
from transformers.models.bert import BertConfig, BertForSequenceClassification

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaSelfOutput,
    RobertaEncoder,
    RobertaOutput,
    RobertaLayer,
    RobertaSelfAttention,
    RobertaAttention,
    RobertaClassificationHead
)
from transformers.models.roberta import RobertaConfig, RobertaForSequenceClassification

from models.adapter import Adapter as MyAdapter
import torch
import torch.nn as nn
import torch.utils.checkpoint


class MyAdapterSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = MyAdapter(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MyAdapterAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)
        self.output = MyAdapterSelfOutput(config)
        self.self = BertSelfAttention(config, position_embedding_type)


class MyAdapterOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)

        self.adapter = MyAdapter(config)

    def forward(self, hidden_states, input_tensor, **kwargs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MyAdapterLayer(BertLayer):
    def __init__(self, config, ):
        super().__init__(config)
        self.attention = MyAdapterAttention(config)
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = MyAdapterAttention(
                config, position_embedding_type="absolute")
        self.output = MyAdapterOutput(config)


class MyAdapterEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [MyAdapterLayer(config) for _ in range(config.num_hidden_layers)])

class BertAdapterModel(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.encoder = MyAdapterEncoder(config)
        self.init_weights()

class RobertaAdapterModel(RobertaModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.encoder = MyAdapterEncoder(config)
        self.init_weights()


class BertAdapterForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertAdapterModel(config)

        # Initialize weights and apply final processing
        self.post_init()

        # BERT fixed all
        for param in self.bert.parameters():
            param.requires_grad = False

        # BERT adapter
        adaters = \
            [self.bert.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(
                config.num_hidden_layers)]

        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True


class RobertaAdapterForSequenceClassification(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaAdapterModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

        # BERT fixed all
        for param in self.roberta.parameters():
            param.requires_grad = False

        # BERT adapter
        adaters = \
            [self.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(
                config.num_hidden_layers)]

        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True



        