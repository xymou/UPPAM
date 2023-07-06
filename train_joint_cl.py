"""
Structural Pre-training
Use a sequence to represent a user
"""
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
import pickle
from tqdm import tqdm
from nltk import word_tokenize

from datasets import load_dataset,load_from_disk
from uppam.data.seq_dataset import JointSeqDataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from uppam.models.seq_skill import PoliBert, PoliRoberta, JointForPretraining
from uppam.trainers.jointseq_trainer import JointSeqTrainer

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    cl_loss: str = field(
        default="triplet",
        metadata={
            "help": "contrastive learning loss: infoNCE, triplet."
        }
    ) 
 

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )

    eval_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The validation data file (.txt or .csv)."}
    )
    train_disk: Optional[str] = field(
        default=None, 
        metadata={"help": "The processed training data."}
    )
    eval_disk: Optional[str] = field(
        default=None, 
        metadata={"help": "The processed validation data."}
    )    

    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    data: Optional[str] = field(
        default='party', 
        metadata={"help": "methods for sampling statements. [party, politag, all]"}
    )        
   

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
@dataclass
class EvalArguments:
    """
    Arguments for Fine-tuning Downstream Tasks
    """
    task_path: Optional[str] = field(
        default='./PoliEval/data', metadata={"help": "Path for Downstream Task Datasets."}
    )

    eval_seed: Optional[int] = field(
        default=42, metadata={"help": "Random Seed."}
    )
    eval_batch_size: Optional[int] = field(
        default=16, metadata={"help": "Batch Size for Fine-tuning."}
    )
    eval_epochs: Optional[int] = field(
        default=30, metadata={"help": "Epochs for Fine-tuning."}
    )    
    eval_lr: Optional[float] = field(
        default=2e-5, metadata={"help": "Learning Rate for Fine-tuning."}
    )     
    eval_weight_decay: Optional[float] = field(
        default=1e-4, metadata={"help": "Weight Decay for Fine-tuning."}
    ) 
    eval_max_len: Optional[int] = field(
        default=256, metadata={"help": "Max Seq Len for Fine-tuning."}
    )     
    eval_max_num_sent: Optional[int] = field(
        default=16, metadata={"help": "Max Sampled Sentence Number for Fine-tuning."}
    )
    eval_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Path of Tokenizer for Fine-tuning."}
    )    

@dataclass
class OurTrainingArguments(TrainingArguments):
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    sent_emb: bool = field(
        default=True,
        metadata={"help": "use sentence embedding for bill in BCL."}        
    )    
    aggr_emb: bool = field(
        default=False,
        metadata={"help": "aggregate sentence embedding for bill in BCL."}        
    )
    aggr_cl: bool = field(
        default=False,
        metadata={"help": "constrastive learning for bill in BCL"}        
    )
    act: str = field(
        default='both',
        metadata={"help": "indicator sequence used in BCL, choose from [both, general, specific]"}        
    )   
    a_sent_emb: bool = field(
        default=False,
        metadata={"help": "use sentence embedding for legislator1 in BCL."}        
    )    
    a_aggr_emb: bool = field(
        default=True,
        metadata={"help": "aggregate sentence embedding to form role embedding for legislator1 in BCL."}        
    )
    a_aggr_cl: bool = field(
        default=False,
        metadata={"help": "constrastive learning for aggreated embeddings for legislator1 in BCL"}        
    )
    b_sent_emb: bool = field(
        default=False,
        metadata={"help": "use sentence embedding for legislator2 in BCL."}        
    )    
    b_aggr_emb: bool = field(
        default=True,
        metadata={"help": "aggregate sentence embedding to form role embedding for legislator2 in BCL."}        
    )
    b_aggr_cl: bool = field(
        default=False,
        metadata={"help": "constrastive learning for aggreated embeddings for legislator2 in BCL"}        
    )        
    leg_sent_emb: bool = field(
        default=False,
        metadata={"help": "sentence embedding in SCL."}        
    )    
    leg_aggr_emb: bool = field(
        default=True,
        metadata={"help": "aggregate sentence embedding to form role embedding in SCL."}        
    )
    leg_aggr_cl: bool = field(
        default=True,
        metadata={"help": "constrastive learning for aggreated embeddings in SCL"}        
    )
    leg_act: str = field(
        default='both',
        metadata={"help": "indicator sequence used in SCL, choose from [both, general, specific]"}        
    )  


    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from transformers.integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.eval_file is not None:
        data_files["dev"] = data_args.eval_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else "\t")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'mlm_backbone' in model_args.model_name_or_path: # backbone got in phase 1
            encoder = PoliRoberta.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,  
                cl_loss = model_args.cl_loss,
            )

        elif 'roberta' in model_args.model_name_or_path:
            encoder = PoliRoberta.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,  
                cl_loss = model_args.cl_loss,
            )
            # Initialize the additional FFNN
            print('Initialize the FFNN Layers...')
            state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
            with torch.no_grad():
                for i in range(12):
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_text.dense.weight=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_text.dense.bias=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_user.dense.weight=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_user.dense.bias=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
        elif 'bert' in model_args.model_name_or_path or 'result' in model_args.model_name_or_path:
            encoder = PoliBert.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,
                cl_loss = model_args.cl_loss,
            )
            print('Initialize the FFNN Layers...')
            state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
            with torch.no_grad():
                for i in range(12):
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_text.dense.weight=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_text.dense.bias=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_user.dense.weight=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_user.dense.bias=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias']) 
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        encoder = AutoModelForMaskedLM.from_config(config)

    encoder.resize_token_embeddings(len(tokenizer))
    model = JointForPretraining(encoder=encoder, hidden_dropout_prob=0.5, emb_dim=768, weight=1)
    print(sum(torch.numel(parameter) for parameter in model.parameters()))
    # Prepare features
    column_names = datasets["train"].column_names
    
    if len(column_names) == 4:
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
        sent3_cname = column_names[3]   
    elif len(column_names)==2:
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]                  
    else:
        raise NotImplementedError

    def prepare_features(examples):
        if len(column_names)==4:
            total = len(examples[sent0_cname])  
            features = {}
            features['bill'] = examples[sent0_cname]
            features['yea'] = examples[sent1_cname]           
            features['anc'] = examples[sent2_cname]
            features['pos'] = examples[sent3_cname]
        elif len(column_names)==2:
            total = len(examples[sent0_cname])  
            features = {}           
            features['anc'] = examples[sent0_cname]
            features['pos'] = examples[sent1_cname]      
        else:
            raise NotImplementedError                  
        return features     

    if training_args.do_train:
        try:
            print('loading data from disk...')
            train_dataset = load_from_disk(data_args.train_disk)
            dev_dataset = load_from_disk(data_args.eval_disk)
            print('data loaded.')
        except:
            train_dataset = datasets["train"].map(
                prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            dev_dataset = datasets["dev"].map(
                prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )        
            train_dataset.save_to_disk(data_args.train_disk)
            dev_dataset.save_to_disk(data_args.eval_disk) 

    train_dataset = JointSeqDataset(train_dataset, tokenizer = tokenizer,
                    max_len=data_args.max_seq_length)
    dev_dataset = JointSeqDataset(dev_dataset, tokenizer = tokenizer, 
                    max_len=data_args.max_seq_length, mem_sents = train_dataset.mem_sents)

    trainer = JointSeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = dev_dataset,
        tokenizer=tokenizer,
        data_collator=None,
    )
    trainer.model_args = model_args    

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True, eval_params = eval_args)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()    