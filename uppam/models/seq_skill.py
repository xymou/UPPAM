"""
Use a sequence to represent a user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.init as init

import transformers
from transformers import RobertaTokenizer
from uppam.models.encoder.modeling_skill import  BertPreTrainedModel, BertModel, BertLMPredictionHead
from uppam.models.encoder.modeling_skill_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
 
        return x


class Aggregator(nn.Module):
    """
    Head for aggregating sentence representation for user/media representation
    e.g., Average Pooler; LSTM; GCN; Pooling aggregator (can add more...)
    """
    def __init__(self, input_dim, output_dim, aggr_method="mean"):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggr_method = aggr_method
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        if self.aggr_method == "lstm":
            self.lstm = nn.LSTM(input_dim, output_dim, num_layers=2, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data,0.0) 


    def forward(self, inputs):
        # inputs: [batch_size, num_sent, emb_dim]
        # mask: [batch_size, num_sent] 
        # ouput: [batch_size, hidden_dim]
       
        if self.aggr_method == "mean":
            features = inputs.mean(dim=1)
        elif self.aggr_method == "max":
            features = self.fc(inputs)
            features = inputs.max(dim=1)[0]
        elif self.aggr_method == "gcn":
            features = self.fc(inputs.max(dim=1)[0])
        elif self.aggr_method == "lstm":
            features, _ = self.lstm(inputs)
            features = features.mean(dim=1) # avg of all hidden states
        else:
            raise ValueError("Unknown aggr type, expected mean, max, gcn, or lstm, but got {}"
                    .format(self.aggr_method))
        return features

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def mlm_forward(
    cls,
    encoder,
    mlm_input_ids=None,
    mlm_labels=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=True,
    skill='text',
):
    mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
    if mlm_input_ids.size()!=attention_mask.size():
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    
    mlm_outputs = encoder(
        mlm_input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        skill=skill,
    )
    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        
    if not return_dict:
        output = (prediction_scores,) + mlm_outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else mlm_outputs

    return MaskedLMOutput(
        loss=masked_lm_loss,
        logits=prediction_scores,
        hidden_states=mlm_outputs.hidden_states,
        attentions=mlm_outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
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
    skill='text',
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        skill=skill,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


def aggr_event_forward(    
    cls,
    encoder,
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
    act = 'both', 
    skill='user',
    ):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    bs = input_ids.size(0)
    num_event = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_event , len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_event, len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_event, len)
    
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        skill=skill,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    pooler_output = pooler_output.view(bs, num_event, pooler_output.size(-1))  # bs * num_event * emb_dim
    if act=='both':
        general_pooler_output = pooler_output[:,0,:] #cls.fc1(pooler_output[:,0,:]) # bs * emb_dim
        specific_pooler_output = pooler_output[:,1,:] #cls.fc2(pooler_output[:,1,:]) # bs * emb_dim
        pooler_output = (general_pooler_output+specific_pooler_output)/2 # bs * emb_dim
    elif act=='general':
        pooler_output = pooler_output[:,0,:] #cls.fc1(pooler_output[:,0,:])
    elif act=='specific':
        pooler_output = pooler_output[:,1,:] #cls.fc2(pooler_output[:,1,:])
    else:
        raise NotImplementedError
  
    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )



def aggr_event_cl_forward(
    cls,
    encoder,
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
    mlm_input_ids=None,
    mlm_attention_mask = None,
    mlm_labels=None,    
    act='both',
    skill='user',
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    bs = input_ids.size(0)
    # Number of roles in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_role = input_ids.size(1)
    num_event = input_ids.size(2)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_role * num_event, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_role * num_event, len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_role* num_event, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        skill=skill,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_attention_mask = mlm_attention_mask.view((-1, mlm_attention_mask.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=mlm_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            skill=skill,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    pooler_output = pooler_output.view(bs*num_role, num_event, pooler_output.size(-1))
    if act=='both':
        general_pooler_output = pooler_output[:,0,:] # cls.fc1(pooler_output[:,0,:]) # bs*num_role * emb_dim
        specific_pooler_output = pooler_output[:,1,:] #cls.fc2(pooler_output[:,1,:]) # bs*num_role * emb_dim
        pooler_output = (general_pooler_output+specific_pooler_output)/2 # bs*num_role * emb_dim
    elif act=='general':
        pooler_output = pooler_output[:,0,:] # cls.fc1(pooler_output[:,0,:])
    elif act=='specific':
        pooler_output = pooler_output[:,1,:] # cls.fc2(pooler_output[:,1,:])
    else:
        raise NotImplementedError

    pooler_output = pooler_output.view(bs, num_role, pooler_output.size(-1)) # bs *num_role * emb_dim

    # Separate representation
    z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z3_list[dist.get_rank()] = z3
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        z3 = torch.cat(z3_list, 0)

    loss_fct = nn.TripletMarginLoss(margin=1.0)
    loss = loss_fct(z1, z2, z3)    
    mlm_loss_fct = nn.CrossEntropyLoss()

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = mlm_loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (pooler_output,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=pooler_output,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class PoliBert(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.cl_loss = model_kargs["cl_loss"]

        cl_init(self, config)

    def forward(self,
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
        sent_emb=False,
        mlm_input_ids=None,
        mlm_attention_mask=None,
        mlm_labels=None,
        act='both',
        only_mlm = None,
        aggr_emb=False,
        aggr_cl=False,
        skill='text',
    ):
        if only_mlm is not None:
            return mlm_forward(self,
                self.bert,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                skill=skill,
            )        
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                skill=skill,
            )
        elif aggr_emb:
            if aggr_cl:
                return aggr_event_cl_forward(self, self.bert,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    mlm_input_ids=mlm_input_ids,
                    mlm_attention_mask=attention_mask,
                    mlm_labels=mlm_labels,   
                    act =act, 
                    skill=skill,
                )
            else:
                return aggr_event_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                act=act,
                skill=skill,
                )                



class PoliRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.cl_loss = model_kargs["cl_loss"]

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
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
        sent_emb=False,
        mlm_input_ids=None,
        mlm_attention_mask=None,
        mlm_labels=None,
        act='both',
        only_mlm=None,
        aggr_emb=False,
        aggr_cl=False,
        skill='text'
    ):
        if only_mlm is not None:
            return mlm_forward(self,
                self.roberta,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                skill=skill,
            )        
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                skill=skill,
            )
        elif aggr_emb:
            if aggr_cl:
                return aggr_event_cl_forward(self, self.roberta,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    mlm_input_ids=mlm_input_ids,
                    mlm_attention_mask=attention_mask,
                    mlm_labels=mlm_labels,   
                    act =act, 
                    skill=skill,
                )
            else:
                return aggr_event_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                act=act,
                skill=skill,
                )  



class JointForPretraining(nn.Module):
    def __init__(self, encoder, **model_kargs):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(model_kargs['hidden_dropout_prob'])
        self.weight = model_kargs['weight']

    def leg_forward(self,
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
        mlm_input_ids=None,
        mlm_attention_mask = None,
        mlm_labels=None,
        sent_emb=False,
        aggr_emb = False,
        aggr_cl = False,
        act='both',     
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_attention_mask = mlm_attention_mask,
            mlm_labels=mlm_labels,
            sent_emb = sent_emb,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,
            act=act,    
            skill='user',         
        )
        

    def vote_forward(self, 
        input_ids=None,
        attention_mask=None,
        sent_emb=True,
        aggr_emb = False,
        aggr_cl = False,        
        a_input_ids=None,
        a_attention_mask=None,
        a_token_type_ids=None,
        a_position_ids=None,
        a_head_mask=None,
        a_inputs_embeds=None,
        a_output_attentions=None,
        a_output_hidden_states=None,
        a_return_dict=None,
        a_sent_emb=False,
        a_aggr_emb = False,
        a_aggr_cl = False,   
        b_input_ids=None,
        b_attention_mask=None,
        b_token_type_ids=None,
        b_position_ids=None,
        b_head_mask=None,
        b_inputs_embeds=None,
        b_output_attentions=None,
        b_output_hidden_states=None,
        b_return_dict=None,
        b_sent_emb=False,
        b_aggr_emb = False,
        b_aggr_cl = False, 
        labels = None,
        act = 'both',
        mlm_input_ids = None,
        mlm_attention_mask = None,
        mlm_labels = None
    ):
        bill_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sent_emb=sent_emb,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,  
            skill='text',        
        )
        bill_pooled_output = bill_outputs.pooler_output

        a_outputs = self.encoder(
            input_ids=a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=a_position_ids,
            head_mask=a_head_mask,
            inputs_embeds=a_inputs_embeds,
            output_attentions=a_output_attentions,
            output_hidden_states=a_output_hidden_states,
            return_dict=a_return_dict,
            sent_emb=a_sent_emb,
            aggr_emb = a_aggr_emb,
            aggr_cl = a_aggr_cl,
            act = act,
            skill='user',
        )
        a_pooled_output = a_outputs.pooler_output

        b_outputs = self.encoder(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            token_type_ids=b_token_type_ids,
            position_ids=b_position_ids,
            head_mask=b_head_mask,
            inputs_embeds=b_inputs_embeds,
            output_attentions=b_output_attentions,
            output_hidden_states=b_output_hidden_states,
            return_dict=b_return_dict,
            sent_emb=b_sent_emb,
            aggr_emb = b_aggr_emb,
            aggr_cl = b_aggr_cl,
            act = act,
            skill='user',
        )
        b_pooled_output = b_outputs.pooler_output

        loss_fct = nn.TripletMarginLoss(margin=1.0)
        loss = loss_fct(bill_pooled_output, a_pooled_output, b_pooled_output)        

        if mlm_input_ids is not None:
            masked_lm_loss = self.encoder(
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                attention_mask=mlm_attention_mask,
                return_dict=True,
                only_mlm=True,
                skill='user',
            ).loss
            loss = loss + self.encoder.model_args.mlm_weight * masked_lm_loss        

        return SequenceClassifierOutput(
            loss=loss,
            logits=bill_pooled_output
        )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        sent_emb=True,
        aggr_emb = False,
        aggr_cl = False,       
        a_input_ids=None,
        a_attention_mask=None,
        a_return_dict=None,
        a_sent_emb=False,
        a_aggr_emb = False,
        a_aggr_cl = False,      
        b_input_ids=None,
        b_attention_mask=None,
        b_return_dict=None,
        b_sent_emb=False,
        b_aggr_emb = False,
        b_aggr_cl = False,
        labels = None,
        act = 'both',
        mlm_input_ids = None,
        mlm_attention_mask = None,
        mlm_labels = None,
        leg_input_ids=None,
        leg_attention_mask=None,
        leg_labels=None,
        return_dict=None,
        leg_mlm_input_ids=None,
        leg_mlm_attention_mask = None,
        leg_mlm_labels=None,
        leg_sent_emb=False,
        leg_aggr_emb = False,
        leg_aggr_cl = False,
        leg_act='both', 
    ):
        vote_outputs = self.vote_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sent_emb= sent_emb,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,       
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
            a_return_dict=a_return_dict,
            a_sent_emb=a_sent_emb,
            a_aggr_emb = a_aggr_emb,
            a_aggr_cl = a_aggr_cl,    
            b_input_ids=b_input_ids,
            b_attention_mask=b_attention_mask,
            b_return_dict=b_return_dict,
            b_sent_emb=b_sent_emb,
            b_aggr_emb = b_aggr_emb,
            b_aggr_cl = b_aggr_cl,
            labels = labels,
            act = act,
            mlm_input_ids = mlm_input_ids,
            mlm_attention_mask = mlm_attention_mask,
            mlm_labels = mlm_labels
        )
        leg_outputs = self.leg_forward(
            input_ids=leg_input_ids,
            attention_mask=leg_attention_mask,
            return_dict=return_dict,
            labels = leg_labels,
            mlm_input_ids=leg_mlm_input_ids,
            mlm_attention_mask = leg_mlm_attention_mask,
            mlm_labels=leg_mlm_labels,
            sent_emb =leg_sent_emb,
            aggr_emb = leg_aggr_emb,
            aggr_cl = leg_aggr_cl,
            act=leg_act,              
        )

        loss = vote_outputs.loss + self.weight*leg_outputs.loss
        return SequenceClassifierOutput(
            loss=loss,
            logits=vote_outputs.logits
        )
