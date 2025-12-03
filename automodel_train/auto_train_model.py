from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext
import torch
import torch.nn as nn
import math
import time
import inspect
import torch.optim as optim
import bitsandbytes as bnb
from dataclasses import dataclass

@dataclass
class RunConfig:
    out_dir = "test-dir-tra"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    lr = 1e-4
    eval_iters = 10
    log_interval = 10
    warmup_iters = 5
    max_steps = 100
    lr_decay_iters = 5000
    min_lr = 1e-5
    beta1=0.9
    beta2 = 0.99
    grad_accm = 4
    weight_decay=1e-1
    decay_lr = True
    device='cuda'
    compile = True
    compile_ac = True
    optim_type = 'adamW32'

class FFTrainer(object):
    """
    Full Finetuning Trainer
    """
    def __init__(self, model_name: str, config: RunConfig):
        self.model_name = model_name
        self.config = config
        self.device = self.config.device
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.config.dtype == 'float16'))
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast('cuda', dtype=torch.bfloat16)
    
    def _ready_model_spec(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _init_optimizer(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad()}

        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim()<2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)} with {num_nodecay_params} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(
            self.config.beta1, self.config.beta2
        ), **extra_args)
        print(f"Using Fused AdamW | {use_fused}")

    @torch.no_grad # Evaluation Loop or Inference
    def compute_loss(self, X, Y):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                with self.ctx:
                    result = self.model.forward(input_ids=X, labels=Y)
                    loss = result.loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_cosine_lr(self, iter):
        if iter < self.config.warmup_iters:
            return self.config.lr * (iter + 1) / (self.config.warmup_iters + 1)

        if iter > self.config.lr_decay_iters:
            return self.config.min_lr
        
        decay_ratio = (iter - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)
    
    def train_loop(self, n_iters, X, y):
        if self.config.compile:
            print(f"Torch Compile | {self.config.compile}")
            self.model = torch.compile(self.model)

        t0 = time.time()
        self.track_iters = 0
        best_val_loss = 1e9
        for iter_num in range(n_iters):
            lr = self.get_cosine_lr(iter_num) if self.config.decay_lr else self.config.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.track_iters = iter_num
            
            if iter_num % self.config.eval_iters == 0:
                lossess = self.compute_loss(X=X, Y=y)
                print(f"Step {iter_num} | Train loss {lossess['train']} | Val Loss {lossess['val']}")
                if lossess['val'] < best_val_loss:
                    best_val_loss = lossess['val']
                    if iter_num > 0:
                        ckpt = {
                            'model': self.model.state_dict(),
                            'optim': self.optimizer.state_dict(),
                            'val_loss': best_val_loss
                        }
                        print(f'Sacing ckpt to {self.config.out_dir}')
                        torch.save(ckpt, f'ckpt_{self.track_iters}')

            for micro_step in range(self.config.grad_accm):
                with self.ctx:
                    out = self.model.forward(input_ids=X, labels=y)
                    logits, loss = out.logits, out.loss
                    loss = loss / self.config.grad_accm

                self.scaler.scale(loss).backward()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % self.config.log_interval == 0:
                lossf = loss.item() * self.config.grad_accm
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            iter_num += 1

            if iter_num > self.config.max_steps:
                break
            

if __name__ == "__main__":
    trainer = FFTrainer(model_name='HuggingFaceTB/SmolLM2-360M', config=RunConfig())