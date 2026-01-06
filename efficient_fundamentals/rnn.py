# RNN For Language Modeling

import torch
import wandb
import math
import torch.nn as nn
from tqdm import tqdm
import tiktoken
from einops import rearrange
from datasets import load_dataset

config = {
    'EPOCHS': 5,
    'step_per_epoch': 1000000,
    'batch_size': 8
}

dataset = load_dataset(
    'roneneldan/TinyStories',
    streaming=True,
    split='train'
)

dataset_batch = dataset.batch(config['batch_size'])
encoding = tiktoken.encoding_for_model('gpt-3.5')

pad_token = 100276

def get_batch_pack(x_list, tokenizer, start_str, end_str, pad_token=pad_token):
    x_list = [f"{start_str}{text}{end_str}" for text in x_list]
    encoded_batch = tokenizer.encode_batch(x_list, disallowed_special=())
    length_encoded_batch = [len(batch) for batch in encoded_batch]
    max_length = max(length_encoded_batch)
    # print(max_length)
    _ = [batch.extend([pad_token]*(max_length - len(batch))) for batch in encoded_batch]
    return encoded_batch


class lstmLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embed_size = 128
        self.hidden_size = 512
        self.num_layers = 4
        self.n_vocab = 100277
        self.input_embed = nn.Embedding(self.n_vocab, self.input_embed_size)
        self.rnn = nn.LSTM(
            input_size=self.input_embed_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.lm_head = nn.Linear(self.hidden_size, self.n_vocab)
    
    def forward(self, x):
        # print('X: ', x.shape)
        if len(x.size()) < 2:
            x = rearrange(x, '(b t) -> b t', b=1)

        x0 = self.input_embed(x)
        # print("X0:", x0.shape)

        out, _ = self.rnn(x0)
        # print('out:', out.shape)

        logits = self.lm_head(out)
        # print(logits.shape)

        return logits

    def generate(self, tok_in, num_tokens=30, end_tok=None):
        generated_toks = []
        ids = torch.tensor([tok_in]).to('cuda')
        
        for _ in range(num_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_ids], dim=1)
            token = next_ids.item()
            generated_toks.append(token)
            if token == end_tok:
                break
        
        print(encoding.decode(generated_toks))
        return generated_toks   

class rnnLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embed_size = 128
        self.hidden_size = 512
        self.num_layers = 4
        self.n_vocab = 100277
        self.input_embed = nn.Embedding(self.n_vocab, self.input_embed_size)
        self.rnn = nn.RNN(
            input_size=self.input_embed_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.lm_head = nn.Linear(self.hidden_size, self.n_vocab)


    def forward(self, x):
        # print('X: ', x.shape)
        if len(x.size()) < 2:
            x = rearrange(x, '(b t) -> b t', b=1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x0 = self.input_embed(x)
        # print("X0:", x0.shape)

        out, hn = self.rnn(x0, h0)
        # print('out:', out.shape)

        logits = self.lm_head(out)
        # print(logits.shape)

        return logits

    def generate(self, tok_in, num_tokens=30, end_tok=None):
        generated_toks = []
        ids = torch.tensor([tok_in]).to('cuda')
        
        for _ in range(num_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_ids], dim=1)
            token = next_ids.item()
            generated_toks.append(token)
            if token == end_tok:
                break
        
        print(encoding.decode(generated_toks))
        return generated_toks

model = rnnLM().to('cuda')

start_story_tok = "<|fim_prefix|>"
end_story_tok = "<|fim_suffx|>"

EPOCHS = config['EPOCHS']

wandb.init(
    project='fundamental-lm',
    config=config
)

@torch.no_grad()
def gradient_norm(params):
    """Computes the 2-norm of the gradients of the given parameters."""
    total = torch.tensor(0.0)
    for p in params:
        if p.grad is not None:
            total = total.to(p.grad.device)
            total += torch.norm(p.grad, dtype=total.dtype) ** 2
    return torch.sqrt(total)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)

total_steps = config['EPOCHS'] * config['step_per_epoch'] // config['batch_size']

pbar = tqdm(total=total_steps)
for epoch in range(EPOCHS):
    model.train()
    for i, x in enumerate(dataset_batch):
        # Dynamic Padding in a stream to max length
        pack = get_batch_pack(x_list=x['text'], tokenizer=encoding, start_str=start_story_tok, end_str=end_story_tok)
        pack_tensor = torch.tensor(pack, dtype=torch.long).to('cuda')
        
        logits = model.forward(pack_tensor)
        labels = pack_tensor[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(
            {"epoch": epoch+1, "step": i+1, "loss": loss.item(), 'ppl': math.exp(loss.item())}
        )

        pbar.update(1)
        wandb.log(
            {
                'loss/train': loss.item(),
                'epoch': epoch+1,
                'step': i+1
            }
        )

        if (i+1) == config['step_per_epoch']//config['batch_size']:
            break
    
    print('='*100)
    print('SAMPLING TEXTTTT')

    sample_text_for_completion = f"{start_story_tok}One"
    tok_in = encoding.encode(sample_text_for_completion, disallowed_special=())
    model.generate(tok_in=tok_in, end_tok=100260, num_tokens=60)




wandb.finish()