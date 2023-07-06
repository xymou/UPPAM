'''
User-level Bias/Ideology Prediction
'''

from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import json
import pickle
import logging
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from PoliEval.polieval.tools.classifier import ModelForRoleClassification, ModelForTextClassification
from PoliEval.polieval.tools.data import sequserdataset

class BIASSeqEval(object):
    def __init__(self, task_path, params, encoder):
        self.seed = params.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.nclasses = params.nclasses
        self.task_path = task_path
        logging.debug('***** Transfer task : Ideology/Bias Prediction *****\n\n')
        logging.debug('***** Task Path : %s *****\n\n', self.task_path)

        train = self.loadFile(os.path.join(self.task_path, 'train.txt'))
        dev = self.loadFile(os.path.join(self.task_path, 'dev.txt'))
        test = self.loadFile(os.path.join(self.task_path, 'test.txt'))
        self.data = {'train': train, 'dev': dev, 'test': test}   
        self.tokenizer = AutoTokenizer.from_pretrained(params.model_name_or_path)
        self.model = ModelForTextClassification(encoder=encoder, sent_emb_dim=768,
                                                num_class=params.nclasses, hidden_dropout_prob=0.5, ctype='single-label')  

        self.batch_size = params.batch_size
        self.epochs = params.epochs
        self.optimizer = AdamW(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        self.max_len = params.max_len
        self.device = encoder.device
        self.model.to(self.device)


    def loadFile(self, path):
        if 'cong_records' in path or 'congs' in path:
            mem, label = [],[]
            with open('./PoliEval/data/LEG_BIAS/cong_records/objspeaker2text.pkl','rb') as f:
                text = pickle.load(f)
            with open('./PoliEval/data/LEG_BIAS/cong_records/objspeaker2label.pkl','rb') as f:
                labels = pickle.load(f)
            with open(path, 'r') as f:
                for line in f.readlines():
                    speaker_id = line.strip()
                    if speaker_id in text and speaker_id in labels: 
                        mem.append(text[speaker_id])
                        label.append(labels[speaker_id])
            return {'label':label, 'mem':mem}
        elif 'TIMME' in path:
            mem, label = [],[]
            with open('./PoliEval/data/LEG_BIAS/TIMME/objspeaker2text.pkl','rb') as f:
                text = pickle.load(f)
            with open('./PoliEval/data/LEG_BIAS/TIMME/objspeaker2label.pkl','rb') as f:
                labels = pickle.load(f)
            with open(path, 'r') as f:
                for line in f.readlines():
                    speaker_id = line.strip()
                    if speaker_id in text and speaker_id in labels: 
                        mem.append(text[speaker_id])
                        label.append(labels[speaker_id])
            return {'label':label, 'mem':mem}
        elif 'PEM' in path:
            mem, label = [],[]
            with open('./PoliEval/data/LEG_BIAS/PEM/objspeaker2text.pkl','rb') as f:
                text = pickle.load(f)
            with open('./PoliEval/data/LEG_BIAS/PEM/objspeaker2label.pkl','rb') as f:
                labels = pickle.load(f)
            with open(path, 'r') as f:
                for line in f.readlines():
                    speaker_id = line.strip()
                    if speaker_id in text and speaker_id in labels: 
                        mem.append(text[speaker_id])
                        label.append(labels[speaker_id])
            return {'label':label, 'mem':mem}          
        elif 'reddit' in path:
            mem, label = [],[]
            ltype = path.split('/')[-2]
            ltype='eco'
            with open('./PoliEval/data/LEG_BIAS/reddit/'+ltype+'/'+'objspeaker2text.pkl','rb') as f:
                text = pickle.load(f)
            with open('./PoliEval/data/LEG_BIAS/reddit/'+ltype+'/'+'objspeaker2label.pkl','rb') as f:
                labels = pickle.load(f)
            with open(path, 'r') as f:
                for line in f.readlines():
                    speaker_id = line.strip()
                    if speaker_id in text and speaker_id in labels and len(text[speaker_id]): 
                        # concat
                        mem.append(text[speaker_id])
                        label.append(labels[speaker_id])
            return {'label':label, 'mem':mem}   
        else:
            raise NotImplementedError



    def tokenize(self):
        self.train_loader = DataLoader(sequserdataset(self.data['train'], self.tokenizer, self.max_len), shuffle=True, batch_size=self.batch_size)
        self.dev_loader = DataLoader(sequserdataset(self.data['dev'], self.tokenizer, self.max_len), shuffle=False, batch_size=self.batch_size)
        self.test_loader = DataLoader(sequserdataset(self.data['test'], self.tokenizer, self.max_len), shuffle=False, batch_size=self.batch_size) 

    def train_epoch(self, epoch):
        self.model.train()
        loop = tqdm(self.train_loader, leave=True)
        for batch in loop:
            self.optimizer.zero_grad()
            mem_input_ids = batch['mem_input_ids'].to(self.device)
            mem_attention_mask = batch['mem_attention_mask'].to(self.device)     
            labels = batch['labels'].to(self.device)      
            outputs = self.model(
                input_ids = mem_input_ids,
                attention_mask = mem_attention_mask,
                sent_emb = True,
                labels = labels,
                return_dict=True,
                skill='user'
            )
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item()) 

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                mem_input_ids = batch['mem_input_ids'].to(self.device)
                mem_attention_mask = batch['mem_attention_mask'].to(self.device)             
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids = mem_input_ids,
                    attention_mask = mem_attention_mask,
                    sent_emb = True,
                    labels = labels,
                    return_dict=True,
                    skill='user'
                )
                preds = torch.argmax(outputs.logits, -1)
                all_preds.append(preds)
                all_labels.append(labels)
            all_labels=torch.cat(all_labels, dim=0)
            all_preds=torch.cat(all_preds, dim=0)
            # print('label:', all_labels, 'pred:', all_preds)
            acc=accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
            f1_macro = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
            return acc, f1_macro

    def run(self):
        # train and evaluate
        # return evaluation results 
        self.tokenize()
        best_f1=0
        best_model = self.model.state_dict()
        counter = 0
        for epoch in range(self.epochs):
            logging.debug('***** Epoch: %d *****\n', epoch)
            self.train_epoch(epoch)
            dev_acc, dev_f1 = self.evaluate(self.dev_loader)
            if dev_f1>best_f1:
                best_f1 = dev_f1
                best_model = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter+=1
                if counter>5:
                    print('Early Stopping...')
                    break 

        self.model.load_state_dict(best_model)
        dev_acc, dev_f1 = self.evaluate(self.dev_loader)
        test_acc, test_f1= self.evaluate(self.test_loader)

        logging.debug('\nDev acc : {0} Dev f1:{1} Test acc : {2} Test f1: {3} for \
            Bias Detection {4} \n'.format(dev_acc, dev_f1, test_acc, test_f1, self.task_path))

        return {'dev_acc': dev_acc, 'dev_f1': dev_f1, 
                'test_acc': test_acc, 'test_f1': test_f1} 