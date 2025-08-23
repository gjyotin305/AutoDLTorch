from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import inspect
import time
import math

@dataclass
class RunConfig:
    model_name: str
    out_dir = "test-dir"
    eval_interval = 250
    eval_iters = 200
    eval_only: bool = False
    log_interval = 10
    gradient_accm_steps: int = 5*8
    batch_size: int = 2
    block_size: int = 64
    always_save_ckpt: bool = True
    wandb_log=False
    wandb_project = 'test'
    wandb_run_name = 'smollm'
    n_layer = 4
    grad_clip: float = 1.0
    n_head = 4
    n_embed = 128
    bias = False
    dropout = 0.0
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    lr = 1e-4
    max_iters = 5000
    lr_decay_iters = 5000
    min_lr = 1e-5
    beta2 = 0.99
    weight_decay = 1e-1
    beta1 = 0.9
    decay_lr = True
    warmup_iters = 100
    device = 'cuda'
    compile: bool = True

class Trainer:
    def __init__(self, config: RunConfig):
        self.config = config
        self.device_type = 'cuda' if 'cuda' in config.device else 'cpu'
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast('cuda', dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(self.device_type)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.config.dtype == 'float16'))
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._init_optimizer(weight_decay=self.config.weight_decay, lr=self.config.lr, betas=(self.config.beta1, self.config.beta2), device=self.config.device)

    def _init_optimizer(self, weight_decay, lr, betas, device):
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)} with {num_nodecay_params} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"Using Fused AdamW {use_fused}")

    
    def get_batch(self, split):
        if split == 'train':
            data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap('test.bin', dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size, ))
        x = torch.stack([torch.from_numpy(data[i: i+ self.config.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1: i+1+self.config.block_size].astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(device=self.config.device, non_blocking=True), y.pin_memory().to(device=self.device_type, non_blocking=True)
        else:
            x, y = x.to(device=self.config.device), y.to(device=self.config.device)
        
        return x, y

    @torch.no_grad # Evaluation Loop or Inference
    def compute_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    result = self.model.forward(input_ids=X, labels=Y)
                    loss = result.loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # to make compatible with other lr schedulers as well
    def get_cosine_lr(self, iter):
        if iter < self.config.warmup_iters:
            return self.config.lr * (iter + 1)/ (self.config.warmup_iters + 1)
        
        if iter > self.config.lr_decay_iters:
            return self.config.min_lr
    
        decay_ratio = (iter - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)


    def train_loop(self, n_iters):
        if self.config.compile:
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)
        X, y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        best_val_loss = 1e9
        for iter_num in range(n_iters):
            lr = self.get_cosine_lr(iter_num) if self.config.decay_lr else self.config.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % self.config.eval_interval == 0:
                losses = self.compute_loss()
                print(f"Step {iter_num}: train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")
                if losses["val"] < best_val_loss or self.config.always_save_ckpt:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'best_val_loss': best_val_loss 
                        }            
                        print(f"Saving ckpt to {self.config.out_dir}")
                        torch.save(checkpoint, f'ckpt_{iter_num}.pt')
                
            if iter_num == 0 and self.config.eval_only:
                break

            for micro_step in range(self.config.gradient_accm_steps):
                with self.ctx:
                    out = self.model.forward(input_ids=X, labels=y)
                    logits, loss = out.logits, out.loss
                    loss = loss / self.config.gradient_accm_steps
                
                X, y = self.get_batch('train')
                self.scaler.scale(loss).backward()
            
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % self.config.log_interval == 0:
                lossf = loss.item() * self.config.gradient_accm_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            iter_num += 1

            if iter_num > self.config.max_iters:
                break

                

if __name__ == "__main__":
    # Test trainer

    trainer = Trainer(config=RunConfig(model_name="HuggingFaceTB/SmolLM2-360M"))
    x, y = trainer.get_batch(split='train')
    
    print(x.shape, y.shape)
    print(x[0], y[0])

    trainer.train_loop(n_iters=100)

   