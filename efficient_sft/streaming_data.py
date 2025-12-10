from datasets import load_dataset
from dataclasses import dataclass
import torch
from einops import rearrange
from transformers import AutoTokenizer

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
        style: str ='conv'
    ) -> None:
        self.streamer_ds = load_dataset(ds_name, streaming=True, split='train')
        self.tokenizer = tokenizer
        self.style = style

    def collator(self, item):
        it_d = InstructionTuningDataset(
            instruction=item['instruction'],
            input=item['input'],
            output=item['output']
        )
        if self.style == 'conv':
            message_in, message_ou = it_d.make_conv_dataset()
        else:
            raise NotImplementedError('Normal Text not implemented')
        
        tokenized_pt = self.tokenizer.apply_chat_template(
            message_in, 
            tokenize=True, 
            return_tensors='pt', 
            add_generation_prompt=False
        )   
        tokenized_pt_label = self.tokenizer.apply_chat_template(
            message_ou, 
            tokenize=True, 
            return_tensors='pt', 
            add_generation_prompt=False
        )
        
        mask_len = len(tokenized_pt[0])
        full_len = len(tokenized_pt_label[0])

        mask_create = [True for _ in range(mask_len)]
        mask_create.extend([False for _ in range(full_len - mask_len)])

        tensor_mask = torch.tensor(mask_create)
        assert len(mask_create) == full_len, "Incorrect Mask"

        input_ids_tensor = tokenized_pt_label[0].clone()
        labels_tensor_masked = tokenized_pt_label[0].masked_fill(tensor_mask, -100)

        input_ids_tensor = rearrange(input_ids_tensor, '(b t) -> b t', b=1)
        labels_tensor_masked = rearrange(labels_tensor_masked, '(b t) -> b t', b=1)

        return {
            'input_ids': input_ids_tensor,
            'labels': labels_tensor_masked,
        }

    def _return_stream_ds(self):
        return self.streamer_ds
    
## Example Usage
# Instantiate tokenzier and StreamingITDataLoader (Streaming Instruction Tuning DataLoader)
# Get IterableDataset Object, apply collator utility function to mask prompt for SFT
# Get {'input_ids': tensor, 'labels': tensor}

# ```python
# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
# dataset = StreamingITDataLoader(ds_name='tatsu-lab/alpaca', tokenizer=tokenizer)
# stream_data = dataset._return_stream_ds()
# for item in stream_data:
#     data = dataset.collator(item)
#     print(data['input_ids'].shape, data['labels'].shape)
# ```