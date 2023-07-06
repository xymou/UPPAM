import torch
import torch.nn as nn
import numpy as np
from nltk import word_tokenize 
from nltk.corpus import stopwords
import random
from tqdm import tqdm
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

import pickle
from nltk import word_tokenize 

with open('./data/raw/frame_indicators.pkl','rb') as f:
    indics = pickle.load(f)
with open('./data/raw/ent_dict.pkl','rb') as f:
    ent_dict = pickle.load(f)
ent_dict= [e.lower() for e in ent_dict]
with open('./data/raw/subj_dict.pkl','rb') as f:
    subj_dict = pickle.load(f)
with open('./data/raw/objpolitical_keywords.pkl','rb') as f:
    poli_words = pickle.load(f)

def get_token_info(texts, indics, ent_dict, subj_dict):
    res = []
    for text in texts:
        tmp = []
        last_loc = {}
        words = word_tokenize(text)
        # frame & sentiment word
        for word in words:
            if word in indics or word in ent_dict:
                if word not in last_loc:
                    loc = text.index(word)
                else:
                    loc = text.index(word, last_loc[word]+1)
                last_loc[word] = loc
                tmp.append((loc, loc+len(word)))
        # entity
        for ent in ent_dict:
            if ent in text:
                loc = text.index(ent)
                tmp.append((loc, loc+len(ent)))
        res.append(tmp)
    return res

class JointSeqDataset(torch.utils.data.Dataset):
    def __init__(self, features, tokenizer, max_len=256, mem_sents=None):
        # store encodings internally
        self.features = features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size= len(self.tokenizer)
        self.load()
        self.words, self.ents = self.load_words()
        # load the triplets
        with open('./data/triplets/vote/objneg_pool_dict.pkl','rb') as f:
            self.vote_neg_pool_dict = pickle.load(f) 
        with open('./data/triplets/party/objneg_pool_dict.pkl','rb') as f:
            self.leg_neg_pool_dict = pickle.load(f) 
        with open('./data/triplets/spon/objneg_pool_dict.pkl','rb') as f:
            self.leg_neg_pool_dict.update(pickle.load(f))            
        if mem_sents is not None:
            self.mem_sents = mem_sents
        else:
            self.prepare()

    def load_words(self):
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
        with open('./data/raw/objmem_tweet_pa_all.pkl','rb') as f:
            self.mem_tweet = pickle.load(f)           

    def prepare(self):
        # prepare indicator sequence for each legislator in each policy area
        self.mem_sents = {}
        mem_tweet = self.mem_tweet
        mem_idx = list(mem_tweet.keys())
        pa = list(mem_tweet[list(mem_tweet.keys())[0]].keys())
        pa.append('other')
        for m in mem_idx:
            self.mem_sents[m]={}
        for m in tqdm(mem_idx):
            for e in pa:
                mem = mem_tweet[m][e]
                num_sent = len(mem)
                if not num_sent:continue
                vectorizer = TfidfVectorizer(stop_words='english',vocabulary=list(set(self.words+self.ents)))
                X = vectorizer.fit_transform(mem)
                data = {'word': vectorizer.get_feature_names(),
                        'tfidf': X.sum(0).tolist()[0]}
                df = pd.DataFrame(data)
                df=df.sort_values(by="tfidf" , ascending=False).reset_index(drop=True)
                df = df[df['tfidf']>0]
                sent = list(df['word'])
                sent = ' '.join(sent[:512])   
                frame_info = get_token_info([sent], indics, ent_dict, subj_dict)[0]
                sent = self.tokenizer(
                        sent,
                        padding = "max_length",
                        max_length = self.max_len,
                        add_special_tokens=True, 
                        truncation=True,
                        return_offsets_mapping=True,            
                    )
                token_span = sent["offset_mapping"]
                mem_input_ids = torch.tensor(sent['input_ids'], dtype=torch.long)
                mem_attention_mask = torch.tensor(sent['attention_mask'], dtype=torch.long)                             
                self.mem_sents[m][e] = {'input_ids': mem_input_ids, 'attention_mask': mem_attention_mask}  
            for t in mem_idx:
                self.mem_sents[t]=self.mem_sents[m]
            break

    def __len__(self):
        # return the number of samples
        return len(self.features)

    def __getitem__(self, i):
        # vote contrastive learning
        bill = self.bill2text[self.features[i]['bill']]
        pos_id = self.features[i]['yea']
        # neg_id = self.features[i]['nay']
        neg_id = random.choice(self.vote_neg_pool_dict[self.features[i]['bill']])
        pa = self.bill2pa[self.features[i]['bill']]

        # general + specific
        pa = ['other', pa]
        if len(pa)==1:
            pa.append('other')

        bill_encoding = self.tokenizer(
            bill,
            padding = "max_length",
            max_length = self.max_len,
            add_special_tokens=True, 
            truncation=True            
        )

        a_input_ids,a_attention_mask=[],[]
        b_input_ids,b_attention_mask=[],[]
        mlm_input_ids, mlm_attention_mask, mlm_labels = [],[],[]
        for e in pa:
            if e in self.mem_sents[pos_id]:
                a_input_ids.append(self.mem_sents[pos_id][e]['input_ids'].unsqueeze(0))
                a_attention_mask.append(self.mem_sents[pos_id][e]['attention_mask'].unsqueeze(0))
            else:
                a_input_ids.append(self.mem_sents[pos_id]['other']['input_ids'].unsqueeze(0))
                a_attention_mask.append(self.mem_sents[pos_id]['other']['attention_mask'].unsqueeze(0))  
            if e in self.mem_sents[neg_id]:
                b_input_ids.append(self.mem_sents[neg_id][e]['input_ids'].unsqueeze(0))
                b_attention_mask.append(self.mem_sents[neg_id][e]['attention_mask'].unsqueeze(0))
            else:
                b_input_ids.append(self.mem_sents[neg_id]['other']['input_ids'].unsqueeze(0))
                b_attention_mask.append(self.mem_sents[neg_id]['other']['attention_mask'].unsqueeze(0))

        yea_input_ids = torch.cat(a_input_ids, dim=0)
        yea_attention_mask = torch.cat(a_attention_mask, dim=0)       
        nay_input_ids = torch.cat(b_input_ids, dim=0)
        nay_attention_mask = torch.cat(b_attention_mask, dim=0)       

        # leg contrastive learning
        input_ids_cat, attention_mask_cat = [], []
        leg_mlm_input_ids, leg_mlm_attention_mask, leg_mlm_labels = [],[],[]
        a_m = self.features[i]['anc']
        p_m = self.features[i]['pos']
        # print(a_m, p_m)
        n_m = random.choice(self.leg_neg_pool_dict[a_m+'_'+p_m])
        # general + specific policy area
        a_input_ids,a_attention_mask=[],[]
        p_input_ids,p_attention_mask=[],[]
        n_input_ids,n_attention_mask=[],[]
        for e in ['other', 'other']:
            if e in self.mem_sents[a_m]:
                a_input_ids.append(self.mem_sents[a_m][e]['input_ids'].unsqueeze(0))
                a_attention_mask.append(self.mem_sents[a_m][e]['attention_mask'].unsqueeze(0))
            else:
                a_input_ids.append(self.mem_sents[a_m]['other']['input_ids'].unsqueeze(0))
                a_attention_mask.append(self.mem_sents[a_m]['other']['attention_mask'].unsqueeze(0)) 
        for e in ['other', 'other']:
            if e in self.mem_sents[p_m]:
                p_input_ids.append(self.mem_sents[p_m][e]['input_ids'].unsqueeze(0))
                p_attention_mask.append(self.mem_sents[p_m][e]['attention_mask'].unsqueeze(0))
            else:
                p_input_ids.append(self.mem_sents[p_m]['other']['input_ids'].unsqueeze(0))
                p_attention_mask.append(self.mem_sents[p_m]['other']['attention_mask'].unsqueeze(0))   
        for e in ['other', 'other']:
            if e in self.mem_sents[n_m]:
                n_input_ids.append(self.mem_sents[n_m][e]['input_ids'].unsqueeze(0))
                n_attention_mask.append(self.mem_sents[n_m][e]['attention_mask'].unsqueeze(0))
            else:
                n_input_ids.append(self.mem_sents[n_m]['other']['input_ids'].unsqueeze(0))
                n_attention_mask.append(self.mem_sents[n_m]['other']['attention_mask'].unsqueeze(0)) 

        a_input_ids = torch.cat(a_input_ids, dim=0)
        a_attention_mask = torch.cat(a_attention_mask, dim=0)     
        p_input_ids = torch.cat(p_input_ids, dim=0)
        p_attention_mask = torch.cat(p_attention_mask, dim=0)   
        n_input_ids = torch.cat(n_input_ids, dim=0)
        n_attention_mask = torch.cat(n_attention_mask, dim=0)   

        input_ids_cat = [a_input_ids.unsqueeze(0), p_input_ids.unsqueeze(0), n_input_ids.unsqueeze(0)]
        attention_mask_cat = [a_attention_mask.unsqueeze(0), p_attention_mask.unsqueeze(0), n_attention_mask.unsqueeze(0)]        
        leg_input_ids = torch.cat(input_ids_cat, dim=0)
        leg_attention_mask = torch.cat(attention_mask_cat, dim=0)   

        return {
            'input_ids':torch.tensor(bill_encoding['input_ids'], dtype=torch.long).flatten(),
            'attention_mask':torch.tensor(bill_encoding['attention_mask'], dtype=torch.long).flatten(),
            'a_input_ids':yea_input_ids, 
            'a_attention_mask':yea_attention_mask,     
            'b_input_ids':nay_input_ids, 
            'b_attention_mask':nay_attention_mask,  
            'leg_input_ids':leg_input_ids,
            'leg_attention_mask': leg_attention_mask,
        }
