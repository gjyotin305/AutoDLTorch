import torch
import wandb
from transformers import AutoTokenizer
from model import FastModel
from tqdm import tqdm
import bitsandbytes as bnb
from torchao.optim import CPUOffloadOptimizer
from streaming_data import StreamingITDataLoader

config = {
    'lr':2e-4,
    'model': 'Full Finetune',
    'dataset': 'Alpaca',
    'steps_epoch': 1000,
    'epochs': 10,
    'grad_accm': 16,
    'grad_ckpt': True  
}

@torch.no_grad()
def gradient_norm(params):
    """Computes the 2-norm of the gradients of the given parameters."""
    total = torch.tensor(0.0)
    for p in params:
        if p.grad is not None:
            total = total.to(p.grad.device)
            total += torch.norm(p.grad, dtype=total.dtype) ** 2
    return torch.sqrt(total)

def setup_optimizer_transformer_model(model, config):
    matrix_params = list(model.model.layers.parameters())
    embedding_params = list(model.model.embed_tokens.parameters())
    lm_head_params = list(model.lm_head.parameters())
    
    # assert len(list(model.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

    adam_groups = [
        dict(params=lm_head_params, lr=2e-4),
        dict(params=matrix_params, lr=2e-5)
    ]

    # optimizer = torch.optim.AdamW(adam_groups)
    # optimizer = bnb.optim.PagedAdamW32bit(adam_groups)
    optimizer = CPUOffloadOptimizer(adam_groups, torch.optim.AdamW, offload_gradients=True, fused=True)
    return optimizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    dataset = StreamingITDataLoader(ds_name='tatsu-lab/alpaca', tokenizer=tokenizer)
    stream_data = dataset._return_stream_ds()
    
    grad_accm_steps = config['grad_accm']
    
    fast = FastModel(model_name='Qwen/Qwen2.5-7B-Instruct', grad_ckpt=config['grad_ckpt'])
    fast.model.train(True)
    
    wandb.init(
        project='efficient-sft',
        config=config
    )

    for p in fast.model.parameters():
        p.requires_grad_(True)
    
    optimizer = setup_optimizer_transformer_model(fast.model, config)
    total_steps = config['epochs'] * config['steps_epoch'] // config['grad_accm']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )
    
    pbar = tqdm(total=total_steps)
    for epoch in range(config['epochs']):
        for idx, item in enumerate(stream_data):
            data = dataset.collator(item)
            
            input_ids = data['input_ids'].to('cuda')
            labels = data['labels'].to('cuda')
            
            out = fast.forward(input_ids=input_ids, labels=labels)
            loss = out['loss'] / grad_accm_steps  
            loss.backward()  

            if (idx + 1) % grad_accm_steps == 0:
                pre_clip = gradient_norm(fast.model.parameters()) 
                torch.nn.utils.clip_grad_norm_(fast.model.parameters(), max_norm=1.0)
                post_clip = gradient_norm(fast.model.parameters())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(
                    {"epoch": epoch, "step": idx//grad_accm_steps, "loss": loss.item()*grad_accm_steps, "pre_grad_norm": pre_clip.item(), "post_grad_norm": post_clip.item()}
                )
                wandb.log(
                    {
                        "loss/train": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "gradient_norm/train": post_clip.item()
                    }
                )
                pbar.update(1)
            
            if (idx+1) % 50 == 0:
                with torch.no_grad():
                    messages = [
                        {"role": "system", "content": "You are an instruction following agent.",},
                        {"role": "user", "content": "## Instruction: Give three tips for staying healthy.\n ## Input: "},
                    ]

                    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

                    _ = fast.generate(
                        tokens=tokens,
                        temperature=0,
                        max_tokens=30
                    )
                

            if (idx+1) == config['steps_epoch']:
                break   
    
    wandb.finish()
    print('Run finished')