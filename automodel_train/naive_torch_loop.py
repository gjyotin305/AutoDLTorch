# Lightning Fabric Rewrite
# import unsloth
from dataclasses import dataclass
import torch
import gdown
from huggingface_hub import HfApi
import os
from tqdm import tqdm
import pandas as pd
import bitsandbytes as bnb
import torchao
from torch.utils.checkpoint import checkpoint
import time
# import deepspeed
# from deepspeed.accelerator import get_accelerator
# from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from torch.profiler import profile, ProfilerActivity, record_function
from model_utils import write_readme_experiment
from torch.utils.data import DataLoader
import torch
from cut_cross_entropy import linear_cross_entropy
from litgpt.utils import chunked_cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Literal
from dataset_new_loader import LMDataset


def apply_grad_ckpt_to_model(model):
    pass


activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

api = HfApi()
scaler = torch.amp.GradScaler()


@dataclass
class RunHyperParams:
    num_epochs: float = 1
    devices: int = 1 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 8
    micro_batch_size: int = 1
    block_size: int = 1024
    grad_clip: float = 1.0
    out_dir: str = 'lit_saves/lora_1'
    warmup_steps: int = 100
    epoch_size: int = 50000

run_config = RunHyperParams()

class CustomModel(torch.nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(run_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        ## KV Caching Fast Model Forward Pass
        pass


def main():

    # fabric = L.Fabric(
    #     accelerator="cuda", 
    #     devices=[0,1], 
    #     strategy='ddp',
    #     precision="bf16-true"
    # )
    # fabric.launch()
    # fabric.seed_everything(1337 + fabric.global_rank)

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', attn_implementation='flash_attention_2', dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    
    # if fabric.global_rank == 0:
    #     os.makedirs(run_config.out_dir, exist_ok=True)

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
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    # setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
    # print("Gradient checkpointing enabled.")
    #  model.gradient_checkpointing_enable()

    model = get_peft_model(peft_config=peft_conf, model=model)
    model.print_trainable_parameters()

    model.to(run_config.device)
    # model = cce_patch(model)
    # model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    # print(model.config.gradient_checkpointing)  # or inspect modules
    # model.prepare_model_for_gradient_checkpointing(model=model)

    # train_loader = fabric.setup_dataloaders(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate)
    # optimizer = torchao.optim.AdamW8bit(model.parameters())
    # optimizer = torchao.optim.CPUOffloadOptimizer(model.parameters(), fused=True)
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=run_config.learning_rate, min_8bit_size=4096, is_paged=False)
    # model, optimizer = fabric.setup(model, optimizer)
    train(model, tokenizer, optimizer, train_loader, run_config.out_dir)
    infer(model, tokenizer)

def infer_standalone(exp_dir):
    model = AutoModelForCausalLM.from_pretrained(exp_dir)
    tokenizer = AutoTokenizer.from_pretrained(exp_dir)

    messages = [
        {"role": "system", "content": "You are an instruction following agent"},
        {"role": "user",   "content": "## Instructions: Describe the life and reign of King Charles II."},
    ]

    # 3. Format the conversation into the model’s input format using the tokenizer’s chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # the `add_generation_prompt=True` tells the model we expect assistant response next. :contentReference[oaicite:3]{index=3}

    # 4. Tokenize and send to model
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # 5. Generate a response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.1
    )

    # 6. Decode & display
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)
    return generated

def infer(model, tokenizer):
    messages = [
        {"role": "system", "content": "You are an instruction following agent"},
        {"role": "user",   "content": "## Instructions: Describe the life and reign of King Charles II."},
    ]

    # 3. Format the conversation into the model’s input format using the tokenizer’s chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # the `add_generation_prompt=True` tells the model we expect assistant response next. :contentReference[oaicite:3]{index=3}

    # 4. Tokenize and send to model
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # 5. Generate a response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.8
    )

    # 6. Decode & display
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)
    return generated


def adapter_save(
    model,
    tokenizer,
    exp_dir,
    repo_name='gjyotin305/sample-lora-qwen'
):
    model.save_pretrained(exp_dir, save_adapter=True, save_config=True)
    tokenizer.save_pretrained(exp_dir)  

    image_output = os.path.join("preview.png")

    gdown.download(id='1P1GGlrrQAyhfoz4pLbGOJSIP3VA2RmZk', output=image_output, quiet=False)

    write_readme_experiment(
        exp_dir=f"{exp_dir}",
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
    
    repo_url = api.create_repo(
        repo_id=f'{repo_name}',  # name of your repo
        private=False,
        repo_type="model",
        exist_ok=True,   # optional
    )
    api.upload_folder(
        folder_path=exp_dir,
        repo_id=f"{repo_name}",
        repo_type="model",
    )


def train(
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
            
        with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                res = model.forward(input_ids=input_ids.to(run_config.device), attention_masks=attention_masks.to(run_config.device), labels=labels.to(run_config.device), output_hidden_states=True)
                logits = res.logits
                # print(logits.device)
                # print(labels.device)
                loss = chunked_cross_entropy(logits[..., :-1, :], labels[..., 1:], chunk_size=128)
                # loss = res.loss
            # print(res[0])
            # print(len(res))
            # print(res.hidden_states)
            # print(len(res.hidden_states))
            # print(model.lm_head.weight.shape)
            # loss = linear_cross_entropy(e=res.hidden_states[-1], c=model.lm_head.weight, targets=labels, shift=True, impl='cce')
            # print(loss.requires_grad)
            # print(loss)
            scaler.scale(loss).backward()

            
            with record_function('optimizer'):
                if (iter_num + 1) % grad_accm_steps == 0:
                    
                    if run_config.grad_clip != 0.0:
                        # scaler.unscale_(optimizer) use if mixed precision
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
        dt = time.time() - d0

        if (iter_num+1) % run_config.log_interval == 0:
            print(f'Step: {iter_num+1} | loss: {loss.item()} | time: {dt}')

        iter_num += 1

        if iter_num == max_iters:
            break
        
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_memory_usage', row_limit=10))
    
    prof.export_chrome_trace('trace.json')
    # adapter_save(
    #     model,
    #     tokenizer,
    #     exp_dir=out_dir
    # )

    
    # merge_adapter(out_dir)

# def merge_adapter(lora_adapter):
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=lora_adapter, 
#     )
#     tokenizer.save_pretrained(lora_adapter)
#     model.push_to_hub_merged(f"gjyotin305/check_litenv_merged", tokenizer, save_method='merged_16bit')

if __name__ == "__main__":
    main()
    # infer_standalone('./lit_saves/lora_1')
    # merge_adapter(run_config.out_dir)