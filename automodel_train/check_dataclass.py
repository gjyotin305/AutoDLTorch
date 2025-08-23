from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass


check = load_dataset('tatsu-lab/alpaca')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

@dataclass
class InstructionDatasetConv:
    instruction: str
    input: str
    output: str
    
    def make_conv_dataset(self) -> str:
        messages = [
            {"role": "system", "content": "You are an instruction following agent",},
            {"role": "user", "content": f"## Instruction: {self.instruction}\n ## Input: {self.input}"},
            {"role": "assistant", "content": f"## Output: {self.output}"}
        ]
        return messages
    
check = """
<|im_start|>system
You are an instruction following agent.<|im_end|>
<|im_start|>user
## Instruction: Give three tips for staying healthy.
 ## Input: <|im_end|>
<|im_start|>assistant
## Output: 1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. 
2. Exercise regularly to keep your body active and strong. 
3. Get enough sleep and maintain a consistent sleep schedule.<|im_end|>
"""

tokenized = tokenizer([check, check])
print(tokenized)
    

