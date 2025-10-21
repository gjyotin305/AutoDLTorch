# Lightning Fabric Rewrite

from dataclasses import dataclass
import lightning as L
import gdown
from huggingface_hub import HfApi
import os
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import time
# from unsloth import FastLanguageModel
import deepspeed
from model_utils import write_readme_experiment
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Literal
from dataset_new_loader import LMDataset

# class CheckpointedDeepSpeedTransformerLayer(torch.nn.Module):
#     def __init__(self, layer):
#         super().__init__()
#         self.layer = layer
    
#     def forward(self, x, *args, **kwargs):
#         # Use checkpoint for memory-efficient forward pass
#         return deepspeed.checkpoint.checkpoint(
#             self.layer, x, *args, **kwargs, use_reentrant=False
#         )

# def checkpoint_layer_forward(layer):
#     """Wraps a layer's forward function with torch checkpoint."""
#     original_forward = layer.forward

#     def checkpointed_forward(*args, **kwargs):
#         # wrap the original forward in checkpoint
#         return checkpoint(original_forward, *args, **kwargs, use_reentrant=True)

#     layer.forward = checkpointed_forward
#     return layer 

# def prepare_model_for_grad_ckpting(model, strategy: Literal['torch', 'deepspeed']):
#     if strategy == 'torch':
#         for layer in tqdm(model.model.model.layers[:10], desc=f'Applying grad ckpt {strategy}'):
#             checkpoint_layer_forward(layer)
#     elif strategy == 'deepspeed':
#         raise NotImplementedError('DeepSpeed yet to be implemented')
    
#     return model

api = HfApi()

@dataclass
class RunHyperParams:
    num_epochs: float = 1
    devices: int = 1 
    log_interval: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 8
    micro_batch_size: int = 1
    block_size: int = 1024
    out_dir: str = 'lit_saves/lora_1'
    warmup_steps: int = 100
    epoch_size: int = 50000

run_config = RunHyperParams()

def main():

    fabric = L.Fabric(accelerator="cuda", devices=[0], precision="32")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    
    if fabric.global_rank == 0:
        os.makedirs(run_config.out_dir, exist_ok=True)

    train_data = LMDataset(
        tokenizer=tokenizer,
        bin_file_data='./train_alpaca.pt'
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=False
    )

    peft_conf = LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(peft_config=peft_conf, model=model)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, tokenizer, optimizer, train_loader, run_config.out_dir)


def adapter_save(
    model,
    tokenizer,
    exp_dir
):
    model.save_pretrained(exp_dir, save_adapter=True, save_config=True)
    tokenizer.save_pretrained(exp_dir)  

    image_output = os.path.join("preview.png")

    gdown.download(id='1P1GGlrrQAyhfoz4pLbGOJSIP3VA2RmZk', output=image_output, quiet=False)

    write_readme_experiment(
        exp_dir=exp_dir,
        title="gjyotin305 Experiment on LoRA finetuning",
        description='testing',
        metadata={
            "model": "qwen2.5/qwen1.5b-instruct",
            "dataset": "alpaca",
            "max_iters": 100,
            "learning_rate": 1e-3,
            "notes": "Used LoRA peft implementation to save memory"
        },
        yaml_metadata={
            'library_name': 'transformers',
            'license': 'apache-2.0',
            'pipeline_tag': 'text-generation',
            'base_model': ['Qwen/Qwen2.5-1.5B-Instruct'],
        },
        image_path='preview.png'
    )
   
    api.upload_folder(
        folder_path=exp_dir,
        repo_id="gjyotin305/qwen2.5-check-litenv-1",
        repo_type="model",
    )



def train(
    fabric,
    model,
    tokenizer,
    optimizer,
    train_data,
    out_dir
):
    iter_num = 0

    grad_accm_steps = run_config.batch_size // run_config.micro_batch_size
   
    # max_iters = run_config.num_epochs * run_config.epoch_size // run_config.micro_batch_size // run_config.devices
    max_iters = 100

    print(f'Iters : {max_iters}')
    for batch in train_data:

        if iter_num <= run_config.warmup_steps:
            lr = run_config.learning_rate*iter_num/run_config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        d0 = time.time()

        input_ids, labels, attention_masks = batch['input_ids'], batch['labels'], batch['attention_mask']   
        res = model.forward(input_ids=input_ids, attention_masks=attention_masks, labels=labels)
        loss = res.loss
        fabric.backward(loss)

        if (iter_num + 1) % grad_accm_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        dt = time.time() - d0

        if (iter_num+1) % run_config.log_interval == 0:
            fabric.print(f'Step: {iter_num+1} | loss: {loss.item()} | time: {dt}')

        iter_num += 1

        if iter_num == max_iters:
            break
    
    adapter_save(
        model,
        tokenizer,
        exp_dir=out_dir
    )

# def merge_adapter(lora_adapter):
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=lora_adapter, 
#     )
#     tokenizer.save_pretrained(lora_adapter)
#     model.push_to_hub_merged(f"gjyotin305/check_litenv_merged", tokenizer, save_method='merged_16bit')

if __name__ == "__main__":
    main()
    # merge_adapter(run_config.out_dir)