import os                                                                                                                                                                                    
import io
import copy
import json
import pickle
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def save_obj(obj, name):
    with open('obj' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,path=None):
    if path is None:
        with open('obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(path+'obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f) 


class sentdataset(Dataset):
    def __init__(self, data, tokenizer, max_len, stance=False):
        self.text = data['text']
        self.label = data['label']
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stance = stance

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        text = self.text[i]
        label = self.label[i]
        encoding = self.tokenizer(
            text,
            padding = "max_length",
            max_length = self.max_len,
            add_special_tokens=True, 
            truncation=True
        )
        if not self.stance:
            return {
                'input_ids':torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask':torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels':torch.tensor(label, dtype=torch.float32)
            }
        else:
            return {
                'input_ids':torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask':torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels':torch.tensor(label, dtype=torch.long)
            }

# use a sequence to represent a user
class sequserdataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.mem = data['mem']
        self.label = data['label']
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.words, self.ents = self.load_words()
        self.prepare()

    def load_words(self):
        with open('./data/raw/frame_indicators.pkl','rb') as f:
            indics = pickle.load(f)
        with open('./data/raw/subj_dict.pkl','rb') as f:
            subj_dict = pickle.load(f)     
        with open('./data/raw/objpolitical_keywords.pkl','rb') as f:
            poli_words = pickle.load(f)
        with open('./data/raw/ent_dict.pkl','rb') as f:
            ent_dict = pickle.load(f)   
        words = list(set(list(indics.keys())+subj_dict+poli_words))
        ents = list(ent_dict)
        words = [w.lower() for w in words]
        ents = [w.lower() for w in ents]        
        return words, ents

    def prepare(self):
        mem_input_ids_lst, mem_attention_mask_lst, labels = [], [], []
        vocab = sorted(list(set(self.words+self.ents))) 
        for i in tqdm(range(len(self.mem))):
            mem = self.mem[i]
            mem = [t.lower() for t in mem]
            label = self.label[i]
            vectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab)
            X = vectorizer.fit_transform(mem)
            data = {'word': vectorizer.get_feature_names(),
                    'tfidf': X.sum(0).tolist()[0]}
            df = pd.DataFrame(data)
            df=df.sort_values(by="tfidf" , ascending=False).reset_index(drop=True)
            df = df[df['tfidf']>0]
            sent = list(df['word'])
            sent = ' '.join(sent[:512])
            sent = self.tokenizer(
                    sent,
                    padding = "max_length",
                    max_length = self.max_len,
                    add_special_tokens=True, 
                    truncation=True                 
                )
            mem_input_ids = torch.tensor(sent['input_ids'], dtype=torch.long)
            mem_attention_mask = torch.tensor(sent['attention_mask'], dtype=torch.long)
            mem_input_ids_lst.append(mem_input_ids)
            mem_attention_mask_lst.append(mem_attention_mask)
            labels.append(label)
        self.dataset = {
            'mem_input_ids':mem_input_ids_lst,
            'mem_attention_mask':mem_attention_mask_lst,
            'labels':labels
        }
        
    def __len__(self):
        return len(self.mem)

    def __getitem__(self, i):
        mem_input_ids = self.dataset['mem_input_ids'][i]
        mem_attention_mask =  self.dataset['mem_attention_mask'][i]
        label = self.dataset['labels'][i]
        return {
            'mem_input_ids':mem_input_ids, 
            'mem_attention_mask':mem_attention_mask,            
            'labels':torch.tensor(label, dtype=torch.long)
        }   


class seqvotedataset(Dataset):
    def __init__(self, data, tokenizer, encoder, max_len, mem_sents=None):
        self.bills =data
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = encoder.device
        self.max_len = max_len
        self.words, self.ents = self.load_words()
        self.load()
        if mem_sents is not None:
            self.mem_sents = mem_sents
        else:
            self.prepare()
        self.dataset = self.prepare_sample()

    def load_words(self):
        with open('./data/raw/frame_indicators.pkl','rb') as f:
            indics = pickle.load(f)
        with open('./data/raw/subj_dict.pkl','rb') as f:
            subj_dict = pickle.load(f)     
        with open('./data/raw/objpolitical_keywords.pkl','rb') as f:
            poli_words = pickle.load(f)
        with open('./data/raw/ent_dict.pkl','rb') as f:
            ent_dict = pickle.load(f)   
        words = list(set(list(indics.keys())+subj_dict+poli_words))
        ents = list(ent_dict)
        words = [w.lower() for w in words]
        ents = [w.lower() for w in ents]
        return words, ents

    def load(self):
        # load corresponding dataset
        with open('./data/triplets/vote/objbill2text.pkl','rb') as f:
            self.bill2text = pickle.load(f)
        with open('./data/triplets/vote/objbill2pa.pkl','rb') as f:
            self.bill2pa = pickle.load(f)
        # load historical statements
        with open('./data/triplets/vote/objmem_tweet_pa_all.pkl','rb') as f:
            self.mem_tweet = pickle.load(f)  

    def prepare_sample(self):
        with open('./data/triplets/vote/objvote.pkl','rb') as f:
            vote = pickle.load(f)
        bill, mem, label, pa = [], [], [], []
        print('prepare vote samples...')
        for bill_id in tqdm(self.bills):
            for leg in vote[bill_id]:
                if (bill_id in self.bill2text) and (leg in self.mem_sents) and len(self.bill2text[bill_id]) and bill_id in (self.bill2pa) and leg!='B001250': 
                    bill.append(bill_id)
                    mem.append(leg)
                    label.append(vote[bill_id][leg])
                    selected_events = self.bill2pa[bill_id]
                    selected_events = ['other']+[selected_events]
                    pa.append(selected_events)
        
        return {'bill':bill, 'mem':mem, 'label':label, 'pa':pa}


    def prepare(self):
        self.mem_sents = {}
        mem_tweet = self.mem_tweet
        mem_idx = list(mem_tweet.keys())
        pa = list(mem_tweet[list(mem_tweet.keys())[0]].keys())
        pa.append('other')
        vocab = sorted(list(set(self.words+self.ents)))
        for m in mem_idx:
            self.mem_sents[m]={}
        for m in tqdm(mem_idx):
            for e in pa:
                mem = mem_tweet[m][e]
                num_sent = len(mem)
                if not num_sent:continue
                vectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab)
                X = vectorizer.fit_transform(mem)
                data = {'word': vectorizer.get_feature_names(),
                        'tfidf': X.sum(0).tolist()[0]}
                df = pd.DataFrame(data)
                df=df.sort_values(by="tfidf" , ascending=False).reset_index(drop=True)
                df = df[df['tfidf']>0]
                sent = list(df['word'])
                sent = ' '.join(sent[:512])   
                sent = self.tokenizer(
                        sent,
                        padding = "max_length",
                        max_length = self.max_len,
                        add_special_tokens=True, 
                        truncation=True,          
                    )
                mem_input_ids = torch.tensor(sent['input_ids'], dtype=torch.long)
                mem_attention_mask = torch.tensor(sent['attention_mask'], dtype=torch.long)                             
                self.mem_sents[m][e] = {'input_ids': mem_input_ids, 'attention_mask': mem_attention_mask}  


    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, i):
        bill_id = self.dataset['bill'][i]
        leg = self.dataset['mem'][i]
        label = self.dataset['label'][i]
        pa = self.dataset['pa'][i]
        
        # bill encoding
        bill_encoding = self.tokenizer(
                self.bill2text[bill_id],
                padding = "max_length",
                max_length = self.max_len,
                add_special_tokens=True, 
                truncation=True            
                )    
        # mem encoding
        mem_input_ids, mem_attention_mask=[],[]
        for e in pa:
            if e in self.mem_sents[leg]:
                mem_input_ids.append(self.mem_sents[leg][e]['input_ids'].unsqueeze(0))
                mem_attention_mask.append(self.mem_sents[leg][e]['attention_mask'].unsqueeze(0))          
            else:
                mem_input_ids.append(self.mem_sents[leg]['other']['input_ids'].unsqueeze(0))
                mem_attention_mask.append(self.mem_sents[leg]['other']['attention_mask'].unsqueeze(0))                
                            
        mem_input_ids = torch.cat(mem_input_ids, dim=0)
        mem_attention_mask = torch.cat(mem_attention_mask, dim=0)      

        bill_input_ids=torch.tensor(bill_encoding['input_ids'], dtype=torch.long).flatten()
        bill_attention_mask=torch.tensor(bill_encoding['attention_mask'], dtype=torch.long).flatten()
    
        return {'bill_input_ids':bill_input_ids,
                'bill_attention_mask':bill_attention_mask,
                'mem_input_ids':mem_input_ids,
                'mem_attention_mask':mem_attention_mask,
                'labels':label
            }




class seqgradedataset(Dataset):
    def __init__(self, data, issue_text, path, tokenizer, encoder, max_len, mem_sents=None):
        self.mems =data
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = encoder.device
        self.max_len = max_len
        self.issue_text = issue_text
        self.path=path
        self.load(self.path)
        self.words, self.ents = self.load_words()
        if mem_sents is not None:
            self.mem_sents = mem_sents
        else:
            self.prepare()
        self.dataset = self.prepare_sample()

    def load_words(self):
        with open('./data/raw/frame_indicators.pkl','rb') as f:
            indics = pickle.load(f)
        with open('./data/raw/subj_dict.pkl','rb') as f:
            subj_dict = pickle.load(f)     
        with open('./data/raw/objpolitical_keywords.pkl','rb') as f:
            poli_words = pickle.load(f)
        with open('./data/raw/ent_dict.pkl','rb') as f:
            ent_dict = pickle.load(f)   
        words = list(set(list(indics.keys())+subj_dict+poli_words))
        ents = list(ent_dict)
        words = [w.lower() for w in words]
        ents = [w.lower() for w in ents]
        return words, ents

    def load(self, path):
        with open(path+'objspeaker2label.pkl','rb') as f:
            self.speaker2label = pickle.load(f)  
        with open(path+'objspeaker2text.pkl','rb') as f:
            self.speaker2text = pickle.load(f)     
        for key in self.speaker2text:
            assert len(self.speaker2text[key])>0

    def prepare_sample(self):
        bill = [self.issue_text] * len(self.mems)
        label=[]
        for m in self.mems:
            label.append(self.speaker2label[m])
        return {'bill':bill, 'mem':self.mems, 'label':label}


    def prepare(self):
        self.mem_sents = {}
        vocab = sorted(list(set(self.words+self.ents)))
        for m in tqdm(self.mems):
            mem = self.speaker2text[m]
            mem = [t.lower() for t in mem]
            num_sent = len(mem)
            if not num_sent:continue
            vectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab)
            X = vectorizer.fit_transform(mem)
            data = {'word': vectorizer.get_feature_names(),
                    'tfidf': X.sum(0).tolist()[0]}
            df = pd.DataFrame(data)
            df=df.sort_values(by="tfidf" , ascending=False).reset_index(drop=True)
            df = df[df['tfidf']>0]
            sent = list(df['word'])
            sent = ' '.join(sent[:512])   
            sent = self.tokenizer(
                    sent,
                    padding = "max_length",
                    max_length = self.max_len,
                    add_special_tokens=True, 
                    truncation=True,          
            )
            mem_input_ids = torch.tensor(sent['input_ids'], dtype=torch.long)
            mem_attention_mask = torch.tensor(sent['attention_mask'], dtype=torch.long)                             
            self.mem_sents[m] = {'input_ids': mem_input_ids, 'attention_mask': mem_attention_mask}  

    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, i):
        bill = self.dataset['bill'][i]
        leg = self.dataset['mem'][i]
        label = self.dataset['label'][i]
        
        # bill encoding
        bill_encoding = self.tokenizer(
                bill,
                padding = "max_length",
                max_length = self.max_len,
                add_special_tokens=True, 
                truncation=True            
                )    
        # mem encoding
        mem_input_ids, mem_attention_mask=[],[]
        for i in range(2):
            mem_input_ids.append(self.mem_sents[leg]['input_ids'].unsqueeze(0))
            mem_attention_mask.append(self.mem_sents[leg]['attention_mask'].unsqueeze(0))                        
                            
        mem_input_ids = torch.cat(mem_input_ids, dim=0)
        mem_attention_mask = torch.cat(mem_attention_mask, dim=0)      

        bill_input_ids=torch.tensor(bill_encoding['input_ids'], dtype=torch.long).flatten()
        bill_attention_mask=torch.tensor(bill_encoding['attention_mask'], dtype=torch.long).flatten()
    
        return {'bill_input_ids':bill_input_ids,
                'bill_attention_mask':bill_attention_mask,
                'mem_input_ids':mem_input_ids,
                'mem_attention_mask':mem_attention_mask,
                'labels':label
            }