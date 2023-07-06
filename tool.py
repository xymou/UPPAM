import logging
from collections import namedtuple
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, device
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Type, Union
from uppam.models.seq_skill import PoliRoberta
from PoliEval.polieval.tools.data import sequserdataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class UPPAMUserEncoder(object):
    def __init__(self, 
        model_name_or_path,
        device: str = None,
        pooler: str = 'cls',
        temp: float = 0.05,
        mlp_only_train: bool = False,
        sent_emb_dim: int = 768,
        output_dim:int =768,
        ):
        model_args = {
            'model_name_or_path':model_name_or_path,
            'pooler_type':pooler,
            'temp':temp,
            'mlp_only_train':mlp_only_train            
        }
        model_args = namedtuple("model_args", model_args.keys())(*model_args.values())
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = PoliRoberta.from_pretrained(model_name_or_path,
            config=config,
            model_args = model_args,
            sent_emb_dim=768,
            output_dim =768,
            cl_loss=None
            )        
        state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
        tmp_dict={}
        for key in state_dict:
            if key.startswith('encoder'):
                tmp_dict[key[8:]] = state_dict[key]
        self.model.load_state_dict(tmp_dict)  
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pooler = pooler

    def encode(self,
        statements: List[List[str]],
        labels: Union[None,List[int]],
        device: str = None,
        batch_size: int = 16,
        max_len: int = 128,
        normalize_to_unit: bool = False,
        ):
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        if labels is None:
            labels = [-1]*len(statements) # -1 when labels are unknown 
        assert len(statements) == len(labels)

        # convert to indicator sequence
        data = {'mem':statements, 'label':labels}
        dataloader = DataLoader(sequserdataset(data, tokenizer=self.tokenizer, max_len=max_len), shuffle=False, batch_size=batch_size)
        
        embedding_list = []
        self.model.eval()      
        with torch.no_grad():
            for batch in tqdm(dataloader):
                mem_input_ids = batch['mem_input_ids'].to(self.device)
                mem_attention_mask = batch['mem_input_ids'].to(self.device)
                outputs = self.model(
                        input_ids = mem_input_ids,
                        attention_mask = mem_attention_mask,
                        sent_emb = True,
                        labels = labels,
                        return_dict=True,
                        skill='user'
                        )                
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        return embeddings

if __name__=="__main__":
    user1 = ["I don' think Trump is a responsible president.", "Abortion is the right of women."]
    user2 = ["We should stand with the life. #prolife !","Gun rights are constitutional."]
    label = None
    model_name = './ckpt/uppam'
    encoder = UPPAMUserEncoder(model_name)
    print("\n=========Get the user representation through their statements============\n")
    emb = encoder.encode([user1, user2], label)
    print(emb)