
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
import numpy as np
from datasets import load_dataset
from torchinfo import summary

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
from uppam.models.seq_skill import PoliBert, PoliRoberta

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy
# Set path to PoliEval
PATH_TO_SENTEVAL = './PoliEval/'
PATH_TO_DATA = './PoliEval/data/'

# Import PoliEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import PoliEval.polieval as polieval
import numpy as np
from datetime import datetime

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def evaluate(model, eval_senteval_transfer, eval_params):
    # eval on downstream tasks
    if eval_senteval_transfer:
        # Set params for PoliEval
        if eval_params:
            params = {'task_path': eval_params.task_path, 'seed':eval_params.eval_seed, 
                    'batch_size':eval_params.eval_batch_size, 'epochs':eval_params.eval_epochs,
                    'lr':eval_params.eval_lr, 'weight_decay':eval_params.eval_weight_decay, 
                    'max_len':eval_params.eval_max_len, 'max_num_sent':eval_params.eval_max_num_sent,
                    'model_name_or_path':eval_params.eval_tokenizer}
        else:
            params = {'task_path': PATH_TO_DATA, 'seed':42, 'batch_size':16, 'epochs':30,
                    'lr':2e-5, 'weight_decay':1e-4, 'max_len':256, 'max_num_sent':16,
                    'model_name_or_path':'bert-base-uncased'}

        se = polieval.engine.SE(params, model)    
        # tasks = ['PUB_BIAS_PEM','PUB_BIAS_TIMME']
        tasks = eval_params.eval_tasks.split(',')
        
        summary(model=model)
        model.eval()
        results = se.eval(tasks)
        metrics = {}
        for key in results:
            metrics[key] = results[key]
        return metrics  


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
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
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
        default=20, metadata={"help": "Epochs for Fine-tuning."}
    )    
    eval_lr: Optional[float] = field(
        default=2e-5, metadata={"help": "Learning Rate for Fine-tuning."}
    )     
    eval_weight_decay: Optional[float] = field(
        default=1e-4, metadata={"help": "Weight Decay for Fine-tuning."}
    ) 
    eval_warmup_ratio: Optional[float] = field(
        default=0, metadata={"help": "Warmup ratio."}
    )     
    eval_max_len: Optional[int] = field(
        default=256, metadata={"help": "Max Seq Len for Fine-tuning."}
    )     
    eval_max_num_sent: Optional[int] = field(
        default=8, metadata={"help": "Max Sampled Sentence Number for Fine-tuning."}
    )
    eval_tokenizer: Optional[str] = field(
        default='./ckpt/uppam', metadata={"help": "Path of Tokenizer for Fine-tuning."}
    )   
    eval_tasks: Optional[str] = field(
        default='PUB_STANCE_poldeb, LEG_BIAS_cong'
    )                    




@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate downstream tasks (dev) at the end of training.
    ## set eval downstream = True if you want to evaluate during training (for selecting best checkpoints) 
    eval_downstream: bool = field(
        default=False,
        metadata={"help": "Evaluate downstream task dev sets (in validation)."}
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
                from .integrations import is_deepspeed_available

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
    parser = HfArgumentParser((ModelArguments, OurTrainingArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    setup_seed(eval_args.eval_seed)
    print('eval seed: ', eval_args.eval_seed)

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
        if 'uppam' in model_args.model_name_or_path:
            model = PoliRoberta.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,
                cl_loss = None,
            )          
            state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
            tmp_dict={}
            for key in state_dict:
                if key.startswith('encoder'):
                    tmp_dict[key[8:]] = state_dict[key]
            model.load_state_dict(tmp_dict)    
            print('successfully load weight...')        
        elif 'roberta' in model_args.model_name_or_path or 'politics' in model_args.model_name_or_path or 'polibertweet' in model_args.model_name_or_path:
 
            model = PoliRoberta.from_pretrained(    
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim = 768,   
                output_dim =768,    
                cl_loss = None,         
            )     
        elif 'bert' in model_args.model_name_or_path:
            model = PoliBert.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,
                cl_loss = None,
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else: # load our pre-trained model
            model = PoliBert.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                sent_emb_dim=768,
                output_dim =768,
                cl_loss = None,
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())    
    else:
        raise NotImplementedError
        model = AutoModelForMaskedLM.from_config(config)

    if 'uppam' not in model_args.model_name_or_path:
        # initialize the parallel FFNN layers
        print('Initialize the FFNN Layers...')
        state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
        if 'roberta' in model_args.model_name_or_path or 'politics' in model_args.model_name_or_path or 'polibertweet' in model_args.model_name_or_path:
            with torch.no_grad():
                for i in range(12):
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_text.dense.weight=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_text.dense.bias=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_user.dense.weight=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.roberta.encoder.layer.__getattr__(str(i)).intermediate_user.dense.bias=torch.nn.Parameter(state_dict['roberta.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
        else:
            with torch.no_grad():
                for i in range(12):
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_text.dense.weight=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_text.dense.bias=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_user.dense.weight=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight'])
                    model.bert.encoder.layer.__getattr__(str(i)).intermediate_user.dense.bias=torch.nn.Parameter(state_dict['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias'])
               
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(training_args.device)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = evaluate(model, eval_senteval_transfer=True, eval_params = eval_args)
        logger.info(results)
    return results
      


if __name__ == "__main__":
    main()