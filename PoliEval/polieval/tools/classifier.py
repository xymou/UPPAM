"""
Task-specific head
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import sys
sys.path.append('./')
    

class ModelForTextClassification(nn.Module):
    def __init__(self, encoder, **model_kargs):
        super().__init__()
        self.encoder = encoder
        self.num_labels = model_kargs['num_class']
        self.classifier = nn.Linear(model_kargs['sent_emb_dim'], model_kargs['num_class'], bias=True)
        self.dropout = nn.Dropout(model_kargs['hidden_dropout_prob'])
        self.ctype = model_kargs['ctype'] # multi-label or single-label
        assert self.ctype in ['single-label', 'multi-label']
        nn.init.xavier_uniform_(self.classifier.weight.data)     

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
        sent_emb=True,
        mlm_input_ids=None,
        mlm_labels=None,
        aggr_emb = False,
        aggr_cl = False,
        skill='text',
        ):
        if self.ctype == 'single-label':
            return self.single_forward(
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
                sent_emb=sent_emb,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                aggr_emb = aggr_emb,
                aggr_cl = aggr_cl,
                skill=skill
            )
        elif self.ctype == 'multi-label':
            return self.multi_forward(
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
                sent_emb=sent_emb,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                aggr_emb = aggr_emb,
                aggr_cl = aggr_cl,
                skill=skill,
            )

    def single_forward(self, 
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
        sent_emb=True,
        mlm_input_ids=None,
        mlm_labels=None,
        aggr_emb = False,
        aggr_cl = False,
        skill='text',  
    ):
        outputs = self.encoder(
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
            sent_emb=sent_emb,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,   
            skill = skill,      
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )        

    def multi_forward(
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
        sent_emb=True,
        mlm_input_ids=None,
        mlm_labels=None,
        aggr_emb = False,
        aggr_cl = False, 
        skill='text',         
    ):
        outputs = self.encoder(
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
            sent_emb=sent_emb,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,
            skill = skill,             
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.squeeze(1)) 

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )         


class ModelForRoleClassification(nn.Module):
    def __init__(self, encoder, **model_kargs):
        super().__init__()
        self.encoder = encoder
        self.num_labels = model_kargs['num_class']
        self.classifier = nn.Linear(model_kargs['sent_emb_dim'], model_kargs['num_class'], bias=True)
        self.dropout = nn.Dropout(model_kargs['hidden_dropout_prob'])
        nn.init.xavier_uniform_(self.classifier.weight.data)

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
        aggr_emb = False,
        aggr_cl = False,
        aggr_key = None,    
        act='both',      
    ):
        outputs = self.encoder(
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
            sent_emb=sent_emb,
            aggr_emb = aggr_emb,
            aggr_cl = aggr_cl,
            act = act,
            skill='user'
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )        



class ModelForRelationClassification(nn.Module):
    def __init__(self, encoder, **model_kargs):
        super().__init__()
        self.encoder = encoder
        self.num_labels = model_kargs['num_class']
        self.classifier = nn.Linear(model_kargs['sent_emb_dim'], model_kargs['num_class'], bias=True)
        self.dropout = nn.Dropout(model_kargs['hidden_dropout_prob'])
        nn.init.xavier_uniform_(self.classifier.weight.data)

    def forward(self, 
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
        a_aggr_key = None,   
        a_event = False,   
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
        b_aggr_key = None,  
        b_event = False, 
        act = 'both',
        labels = None  
    ):
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
            skill='text',
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
            skill = 'user',
        )
        b_pooled_output = b_outputs.pooler_output

        pooled_output = torch.mul(a_pooled_output, b_pooled_output) #内积
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)        

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )         

