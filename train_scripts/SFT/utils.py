from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch


def generate_prompt(example, eos_token):
    prompt = f"### Instructions: \n{example['instruction']}\n\n ### Input:\n {example['input']} \n\n ### Response:"
    answer = f"{example['output']}{eos_token}"
    return prompt, answer    


class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): 
        example = self.dataset[index]
        prompt, answer = generate_prompt(example, self.tokenizer.eos_token)
            
        input_ids = self.tokenizer.encode(
            prompt + answer,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        mask = len(self.tokenizer.encode(prompt))

        labels = input_ids[0].tolist()
        
        labels[0:mask] = [-100]*mask
        single_labels = torch.tensor([labels])

        single_labels[single_labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids[0],
            "labels": single_labels[0]
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train") 
    max_length = 8192
    supervised_dataset = SFTDataset(dataset, tokenizer, max_length)
    dataloader = DataLoader(supervised_dataset, batch_size=2)

    for x in supervised_dataset:
        print(x)
        print(x['input_ids'].shape, x['labels'].shape)
        break
    
    for x in dataloader:
        print(x)
        print(x['input_ids'].shape, x['labels'].shape)
        break

