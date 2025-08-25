from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

@dataclass
class InstructionDataset:
    instruction: str
    input: str
    output: str
    template: str = """You are an instruction following assistant. \nINSTRUCTION: {instruct}\nINPUT: {input}\nOUTPUT: {output}"""

    def render(self) -> str:
        return self.template.format(instruct=self.instruction, input=self.input, output=self.output)

@dataclass
class InstructionDatasetConv:
    instruction: str
    input: str
    output: str
    
    def make_conv_dataset(self) -> str:
        messages = [
            {"role": "system", "content": "You are an instruction following agent.",},
            {"role": "user", "content": f"## Instruction: {self.instruction}\n ## Input: {self.input}"},
            {"role": "assistant", "content": f"## Output: {self.output}"}
        ]
        return messages


class LMDataset(Dataset):
    def __init__(self, bin_file_data: str, block_size: int = 1024) -> None:
        super().__init__()
        self.bin_file_data = bin_file_data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = torch.load(self.bin_file_data)
        self.block_size = block_size
        self.max_dlen = self.dataset['input_ids'].shape[1]

    def __len__(self):
        return self.dataset['input_ids'].shape[0]
    
    def __getitem__(self, index):
        input_ids = self.dataset['input_ids'][index]
        attention_mask = self.dataset['attention_mask'][index]
        
        # Block_size
        input_ids_fin = input_ids[:self.block_size]
        labels = input_ids[1:self.block_size+1]
        attention_mask = attention_mask[:self.block_size]

        result = {
            'input_ids': input_ids_fin.pin_memory().to(self.device, non_blocking=True),
            'attention_mask':attention_mask.pin_memory().to(self.device, non_blocking=True),
            'labels': labels.pin_memory().to(self.device, non_blocking=True)
        }

        return result


class DataCollator:
    """
    Pad Sequences to Max Length
    """
    def __init__(self, dataset: Dataset, tokenizer_str: str) -> None:
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    
    def process_sequences(self, batches):
        input_ids = [batch['input_ids'] for batch in batches]
        attention_masks = [batch['attention_mask'] for batch in batches]

        max_len = max(len(seq) for seq in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            ids = ids[:max_len]
            mask = mask[:max_len]
            
            padded_length = max_len - len(ids)
            if padded_length > 0:
                ids.extend([self.tokenizer.pad_token_id]*padded_length)
                mask.extend([0]*padded_length)

            padded_input_ids.extend(ids)
            padded_attention_masks.extend(mask)

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long)
        }
    
    def process_sequences_vectorized(self, batches):
        input_ids = [batch['input_ids'] for batch in batches]
        attention_masks = [batch['attention_mask'] for batch in batches]
        
        max_len = max(len(seq) for seq in input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = len(input_ids)
        
        padded_input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
        padded_attention_masks = np.zeros((batch_size, max_len), dtype=np.int64)
        
        for i in tqdm(range(len(input_ids)), total=len(input_ids)):
            ids = input_ids[i]
            mask = attention_masks[i]

            seq_len = min(len(ids), max_len)
            padded_input_ids[i, :seq_len] = ids[:seq_len]
            padded_attention_masks[i, :seq_len] = mask[:seq_len]
   
        return {
            'input_ids': torch.from_numpy(padded_input_ids),
            'attention_mask': torch.from_numpy(padded_attention_masks)
        }
    
    @staticmethod
    def save_torch_tensor(object_tensor, file_path):
        torch.save(object_tensor, f'{file_path}.pt')
        print('Saved')
        print('Check')
        loaded = torch.load(f'{file_path}.pt')
        assert torch.allclose(loaded['input_ids'], object_tensor['input_ids']), "Not Same"
        print("Correct")


class BaseTokenizedDataset(object):
    def __init__(self, tokenizer_model: str, dataset_name: str, prompt_template, split: str, conv_style: bool = True, text_col: str='text_tune') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.dataset = load_dataset(dataset_name)
        self.prompt_template = prompt_template
        self.text_col = text_col
        self.split = split
        self.conv_style = conv_style
        self.get_text_column(split)
        self.tokenized = self._tokenize_dataset()
        
    def process_text(self, examples):
        texts = []
        for instruct, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            text = self.prompt_template(instruction=instruct, input=inp, output=out)
            texts.append(text.render())
        return {f"{self.text_col}": texts}

    def process_text_conversation_style(self, examples):
        conv = []
        
        for instruct, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            text = self.prompt_template(instruction=instruct, input=inp, output=out)
            messages = text.make_conv_dataset()
            text_temp = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            conv.append(text_temp)

        return {f'{self.text_col}': conv}

    def get_text_column(self, split: str='train'): 
        if self.conv_style:
            self.dataset[split] = self.dataset[split].map(
                self.process_text_conversation_style,
                batched=True,
                num_proc=10,
                remove_columns=['text']
            )
        else:
            self.dataset[split] = self.dataset[split].map(
                self.process_text,
                batched=True,
                num_proc=10,
                remove_columns=['text']
            )

        return self.dataset[split]

    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_col])

    def _tokenize_dataset(self):
        tokenized_dataset = self.dataset[self.split].map(
            self.tokenize_function,
            batched=True
        )
        return tokenized_dataset


if __name__ == "__main__":
    # check = BaseTokenizedDataset(
    #     tokenizer_model='Qwen/Qwen2.5-1.5B-Instruct',
    #     dataset_name='tatsu-lab/alpaca',
    #     prompt_template=InstructionDatasetConv,
    #     split='train'
    # )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

    # print(check.dataset['train'])
   
    # print(check.tokenized)
   
    # check_collator = DataCollator(
    #     dataset=check.tokenized,
    #     tokenizer_str='Qwen/Qwen2.5-1.5B-Instruct'
    # )

    # check_tensor = check_collator.process_sequences_vectorized(batches=check.tokenized)
    # print(check_tensor['input_ids'].shape, check_tensor['attention_mask'].shape)
    # check_collator.save_torch_tensor(
    #     object_tensor=check_tensor,
    #     file_path="train_alpaca"
    # )

    dataset = LMDataset(
        bin_file_data="./train_alpaca.pt",
        block_size=1024
    )

    print(tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id)

    for i, (res) in enumerate(dataset):
        x, y, z = res['input_ids'], res['labels'], res['attention_mask']
        print(x.shape, y.shape, z.shape)
        check = x[:100].cpu().numpy().tolist()

        print("="*100)
        print(tokenizer.decode(check))
        print("="*100)

        print(x[:100], y[:100], z[:100])
        if i == 2:
            break

        
