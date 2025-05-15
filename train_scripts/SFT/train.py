from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import torch
import os
import bitsandbytes as bnb
from utils import SFTDataset
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig


config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none"
)

device = "cuda:1"
max_length = 8192
batch_size = 4
n_epochs = 1
micro_batch_size = 1


model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

train_data = load_dataset("yahma/alpaca-cleaned", split="train")

train_dataset = SFTDataset(train_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=2)

model = get_peft_model(model, config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# optimizer = bnb.optim.Adam8bit(model.parameters(), min_8bit_size=16384)

num_training_steps = len(train_loader) * n_epochs // (batch_size // micro_batch_size)
gradient_accml_steps = batch_size // micro_batch_size
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=5,
    num_training_steps=num_training_steps
)

model.train()
accumulated_loss = 0
accumulated_steps = 0
print("Variables Initialized, Training Started")

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    for index, batch in enumerate(tqdm(train_loader)):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        out = model(**batch)
        loss = out.loss
        loss = loss / gradient_accml_steps
        loss.backward()

        accumulated_loss += loss.item()
        accumulated_steps += 1

        if (index + 1) % gradient_accml_steps == 0 or index == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Loss: {accumulated_loss / accumulated_steps}")
            accumulated_loss = 0
            accumulated_steps = 0

    checkpoint_dir = os.path.join("./", f"checkpoint-epoch-{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    print("Model Saved")

print("Training_completed")

