from datasets import load_dataset
import tiktoken
import torch

encoding = tiktoken.encoding_for_model('gpt-3.5')

dataset = load_dataset(
    'roneneldan/TinyStories',
    streaming=True,
    split='train'
)

def get_batch_pack(x_list, tokenizer, start_tok, end_tok, pad_token=-100):
    x_list = [f"{start_tok}{text}{end_tok}" for text in x_list]
    encoded_batch = tokenizer.encode_batch(x_list)
    length_encoded_batch = [len(batch) for batch in encoded_batch]
    max_length = max(length_encoded_batch)
    # print(max_length)
    _ = [batch.extend([pad_token]*(max_length - len(batch))) for batch in encoded_batch]
    return encoded_batch

dataset_batch = dataset.batch(32)

for idx, x in enumerate(dataset_batch):
    # print(len(x['text']))
    pack = get_batch_pack(x_list=x['text'], tokenizer=encoding, start_tok=1, end_tok=0)
    # print(pack)
    pack_tensor = torch.tensor(pack, dtype=torch.long)
    print(pack_tensor.shape)
    
    if idx == 2:
        break