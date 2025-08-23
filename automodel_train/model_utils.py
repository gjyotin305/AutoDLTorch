from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from dataclasses_json import dataclass_json
import os
import numpy as np
import torch
import inspect
import torch.nn as nn

@dataclass_json
@dataclass
class ModelConfig:
    model_name: str
    block_size: str
    n_layer: int
    n_head: int 
    vocab_size: int
    n_embed: int
    dropout: float 


class PrepareData(object):
    def __init__(self, tokenizer: AutoTokenizer, ds_name: str) -> None:
        self.tokenizer = tokenizer
        self.ds_name = ds_name
        self.ds = load_dataset(ds_name, num_proc=128)
        self._process_data()
    
    def _process_data(self):
        if hasattr(self.ds, 'train'):
            self.train_ds = self.ds['train']
        
        if hasattr(self.ds, 'test'):
            self.eval_ds = self.ds['test']
        elif hasattr(self.ds, 'val'):
            self.eval_ds = self.ds['val']
        else:
            self.train_ds = self.ds['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=False)

    def tokenize_process(self, example):
        ids = self.tokenizer.encode(example['text'])
        ids.append(self.tokenizer.eos_token_id)

        return {
            "ids": ids,
            "len": len(ids)
        }
    
    def prepare_dataset(self, total_batch_num: int = 1024):
        tokenized = self.train_ds.map(
            self.tokenize_process,
            remove_columns=['text'],
            desc='tokenizing the splits',
            num_proc=128
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = total_batch_num

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])

                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()



class ModelTrain(object):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._prepare_model()


    def _prepare_model(self):
        self.model_name = self.config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.block_size = self.config.block_size
        self.config.n_head = self.model.config.num_attention_heads
        self.config.vocab_size = self.model.config.vocab_size
        self.config.n_layer = self.model.config.num_hidden_layers
        self.config.n_embed = self.model.config.hidden_size
    

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        for block in self.model.model.layers:
            if hasattr(block.self_attn, 'bias'):
                block.self_attn.bias = block.self_attn.bias[:, :, :block_size, :block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using Fused AdamW {use_fused}")

        return optimizer


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        return n_params


    def set_model_from_scratch(self):
        pass


    def forward(self):
        pass


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    dataloader = PrepareData(tokenizer=tokenizer, ds_name="openwebtext")
    dataloader.prepare_dataset()

    pass