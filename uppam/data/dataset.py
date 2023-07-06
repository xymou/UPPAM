import torch
import torch.nn as nn
import numpy as np
from nltk import word_tokenize 
import random
from tqdm import tqdm
import pickle


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, features, mask_ind_prob = 0.15, mask_token_prob=0, vocab_size=50265):
        # store encodings internally
        self.features = features
        self.mask_ind_prob = mask_ind_prob
        self.mask_token_prob = mask_token_prob
        self.vocab_size = vocab_size

    def __len__(self):
        # return the number of samples
        return len(self.features)

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        # res = {key: tensor[i].clone() for key, tensor in self.features.items() if key in special_keys}
        res = self.features[i]
        special_keys = ['input_ids', 'attention_mask', 'type_token_ids', 'labels']
        res = {key:torch.tensor(value) for key, value in res.items() if key in special_keys}
        token_span = self.features[i]['token_span']
        ind_info = self.features[i]['frame_info']
        input_ids = res['input_ids']
        labels = res['labels']
        # mask indicator
        for i in range(len(ind_info)):
            start, end = ind_info[i] 
            token_pos = []
            for j in range(len(token_span)):
                if token_span[j][0]>= start and token_span[j][1]<=end:
                    token_pos.append(j)
            prob = random.random()
            if prob< self.mask_ind_prob:
                prob /= self.mask_ind_prob
                if prob<0.8:
                    input_ids[token_pos] = 50264
                elif prob<0.9:
                    change_token = random.choices(list(range(self.vocab_size)), k=len(token_pos))
                    input_ids.scatter_(0, torch.LongTensor(token_pos), torch.LongTensor(change_token))
                else:
                    continue
        # randomly mask other tokens 
        rand = torch.rand(input_ids.shape)
        # 80% be [MASK]
        mask_arr = (rand < 0.8* self.mask_token_prob) * (input_ids > 3) 
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        input_ids[selection] = 50264 
        # 10% be changed to a random token
        change_arr = (rand< 0.9 * self.mask_token_prob) * (rand> 0.8* self.mask_token_prob ) * (input_ids > 3) 
        selection = torch.flatten(change_arr.nonzero()).tolist()
        change_token = random.choices(list(range(self.vocab_size)), k=len(selection))
        input_ids.scatter_(0, torch.LongTensor(selection), torch.LongTensor(change_token))                
                
        indices = (input_ids!=50264)
        indices = torch.flatten(indices.nonzero()).tolist()
        labels[indices] = -100
        res['mlm_input_ids'] = input_ids
        res['mlm_labels'] = labels
        res['only_mlm'] = True
        # print(888, res)
        return res

