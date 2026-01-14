from datasets import load_dataset
from dataclasses import dataclass
import torch
from tqdm import trange
from einops import rearrange
from typing import List
from transformers import AutoTokenizer

@dataclass
class ConversationDataset:
    conversations: List
    format_change: bool = False 

    def convert_from_value_to_role_content(self):
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system"
        }

        converted = []
        for msg in self.conversations:
            converted.append({
                "role": role_map.get(msg["from"], msg["from"]),
                "content": msg["value"]
            })

        return converted

    def make_sample(self):
        if self.format_change:
            self.conversations = self.convert_from_value_to_role_content()
        
        assert self.conversations[-1]['role'] == 'assistant'
        return self.conversations[:-1], self.conversations[:]

@dataclass
class InstructionTuningDataset:
    instruction: str
    input: str
    output: str

    def make_dataset(self):
        prompt = f"""## Instruction {self.instruction} \n ## Input {self.input}"""
        output = f"""## Instruction {self.instruction} \n ## Input {self.input} \n ## Output {self.output}"""
        return prompt, output
    
    def make_conv_dataset(self):
        message_input = [
            {"role": "system", "content": "You are an instruction following agent.",},
            {"role": "user", "content": f"## Instruction: {self.instruction}\n ## Input: {self.input}"},
         ]
        message_output = [
            {"role": "system", "content": "You are an instruction following agent.",},
            {"role": "user", "content": f"## Instruction: {self.instruction}\n ## Input: {self.input}"},
            {"role": "assistant", "content": f"## Output: {self.output}"}
         ]
        return message_input, message_output

class StreamingITDataLoader: 
    def __init__(
        self, 
        ds_name: str, 
        tokenizer: AutoTokenizer, 
        style: str ='conv',
        pre_conv: bool = False,
        packing: bool = False,
        format_change: bool = False,    
        pack_size: int = 512,
        batch_size: int = 32
    ) -> None:
        self.streamer_ds = load_dataset(
            ds_name, 
            split='train[100:200]'
        )
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.style = style
        self.packing = packing
        self.format_change = format_change
        self.pre_conv = pre_conv
        self.pack_size = pack_size

    def collator(self, item):
        # Single Stream Inference
        if self.packing is not True:
            if self.pre_conv is not True:
                it_d = InstructionTuningDataset(
                    instruction=item['instruction'],
                    input=item['input'],
                    output=item['output']
                )
                if self.style == 'conv':
                    message_in, message_ou = it_d.make_conv_dataset()
                else:
                    raise NotImplementedError('Normal Text not implemented')
            else:
                it_c = ConversationDataset(
                    conversations=item['conversations'],
                    format_change=self.format_change
                )
                message_in, message_ou = it_c.make_sample()
            
            # Tokenize both
            tokenized_in  = self.tokenizer.apply_chat_template(
                message_in,
                tokenize=True,
                return_tensors='pt',
                add_generation_prompt=False,
            )

            tokenized_lbl = self.tokenizer.apply_chat_template(
                message_ou,
                tokenize=True,
                return_tensors='pt',
                add_generation_prompt=False,
            )

            input_ids = tokenized_lbl[0]

            mask_len = tokenized_in[0].size(0)
            full_len = input_ids.size(0)

            mask = torch.arange(full_len) < mask_len

            # Build labels: original where mask is False, -100 where True
            labels = input_ids.masked_fill(mask, -100)

            # Add batch dimension
            input_ids = input_ids.unsqueeze(0)
            labels     = labels.unsqueeze(0)

            return {
                'input_ids': input_ids,
                'labels': labels,
            }
        elif self.packing is True:
            raise NotImplementedError('Packing not Implemented')
        
            for i in trange(self.batch_size, desc='Batching'):
                it_d = InstructionTuningDataset(
                    instruction=item['instruction'][i],
                    input=item['input'][i],
                    output=item['output'][i]
                )
                if self.style == 'conv':
                    message_in, message_ou = it_d.make_conv_dataset()
                else:
                    raise NotImplementedError('Normal Text not implemented')
                tokenized_in  = self.tokenizer.apply_chat_template(
                    message_in,
                    tokenize=True,
                    return_tensors='pt',
                    add_generation_prompt=False,
                )

                tokenized_lbl = self.tokenizer.apply_chat_template(
                    message_ou,
                    tokenize=True,
                    return_tensors='pt',
                    add_generation_prompt=False,
                )

                input_ids = tokenized_lbl[0]

                mask_len = tokenized_in[0].size(0)
                full_len = input_ids.size(0)

                mask = torch.arange(full_len) < mask_len

                # Build labels: original where mask is False, -100 where True
                labels = input_ids.masked_fill(mask, -100)

                # Add batch dimension
                input_ids = input_ids.unsqueeze(0)
                labels     = labels.unsqueeze(0)

                return_it =  {
                    'input_ids': input_ids,
                    'labels': labels,
                }

    def _return_stream_ds(self):
        return self.streamer_ds
    
## Example Usage
# Instantiate tokenzier and StreamingITDataLoader (Streaming Instruction Tuning DataLoader)
# Get IterableDataset Object, apply collator utility function to mask prompt for SFT
# Get {'input_ids': tensor, 'labels': tensor}

# ```python
# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
#     dataset = StreamingITDataLoader(
#         ds_name='jdoo2/Qwen2.5-32B-Instruct_long_context_range80-100_train_data_100k', 
#         tokenizer=tokenizer, 
#         pre_conv=True,
#         format_change=True
#     )
#     stream_data = dataset._return_stream_ds()
#     # batch_dataset = stream_data.batch(batch_size=32)
#     for item in stream_data:
#         data = dataset.collator(item)
#         print(data['input_ids'].shape, data['labels'].shape)
#         break
# ```