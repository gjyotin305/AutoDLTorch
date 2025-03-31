# gdown --id 1bEK6RCdnXIqg8JGrJIMvDaAM-baalGwt For dataset

from torch.utils.data import (
    DataLoader, 
    Dataset
)
import pandas as pd
import os
import torch
from tqdm import tqdm
import json
from spacy.lang.hi import Hindi
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from typing import Tuple, List

def get_data(file_path: str) -> List[str]:
    with open(file_path) as f:
        lines = [line.strip() for line in f]
    
    return lines

class Vocab:
    def __init__(self, lang: str, file_path: str = None):
        self.lang = lang
        self.file_path = file_path
        self.word2index = {
            "<sos>": '0',
            "<eos>": '1',
            "<unk>": '2',
            "<pad>": '3'
        }
        self.index2word = {
            '0': "<sos>",
            '1': "<eos>",
            '2': "<unk>",
            '3': "<pad>"
        }
        if lang == "en":
            self.tokenizer = Tokenizer(English().vocab)
        elif lang == "hi":
            self.tokenizer = Tokenizer(Hindi().vocab)
        if file_path is not None:
            self._consume()

    def convert_word2index(self, word: str) -> int:
        if word in self.word2index:
            return self.word2index[word]
        else: 
            return self.word2index['<unk>']

    def add_word(self, word: str) -> int:
        if word in self.word2index:
            return self.word2index[word]
    
        idx = len(self.word2index)
        self.word2index[word] = str(idx)
        self.index2word[str(idx)] = word

        return idx

    def _consume(self):
        with open(self.file_path) as f:
            lines = [line.strip() for line in f]
            f.close()

        for line in lines:
            toks = self.tokenizer(line)
            for tok in toks:
                self.add_word(tok.text)
            
        print(f"Index Updated with Vocab of length {len(self.word2index)}")

        with open(f'{self.file_path}_saved_vocab.json', 'w') as f:
            json.dump({
                "word2index": self.word2index,
                "index2word": self.index2word
            }, f, indent=2)
        
        print("Vocabulary saved")
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            data = json.load(f)
            f.close()
        
        self.word2index = data['word2index']
        self.index2word = data['index2word']
        
        print(f'Length of vocabulary {len(self.word2index)}')
        print(f'Length of rv {len(self.index2word)}')

        return 200

    def convert_to_seq(
        self, 
        tokens_str: List[str], 
        verbose: bool = False, 
        length: int = None
    ):
        return_seq = [0]
        return_seq.extend([self.convert_word2index(tok.text) for tok in tokens_str])
        
        if length is not None:
            if length > len(return_seq):
                return_seq.append(1)
                padded_sequence = [3 for i in range(length+2 - len(return_seq))]
                return_seq.extend(padded_sequence)
            elif length < len(return_seq):
                return_seq = return_seq[:length+1]
                return_seq.append(1)
            elif length == len(return_seq):
                return_seq.append(1)
                return_seq.append(3)


            assert len(return_seq) == length+2, f"Actual Length {len(return_seq)}" # <eos> + <sos> account for 2 tokens
        elif length is None:
            return_seq.append(1)

        if verbose is True:
            check = [self.index2word[str(tok)] for tok in return_seq]
            print(check)

        return return_seq

class English2HindiData(Dataset):
    def __init__(
        self, 
        folder_path: str, 
        split: str,
        max_length: int,
        vocab_en: Vocab, 
        vocab_hi: Vocab
    ):
        super().__init__()
        self.folder_path = folder_path
        self.split = split
        self.max_length = max_length
        self.vocab_en = vocab_en
        self.vocab_hi = vocab_hi
        self.en_tokenizer = Tokenizer(English().vocab)
        self.hi_tokenizer = Tokenizer(Hindi().vocab)
        self._consume_text()
    
    def __len__(self):
        assert len(self.en_data) == len(self.hi_data)
        return len(self.en_data)

    def _find_files(self):
        number_of_files = os.listdir(self.folder_path)
        
        valid_files = []
        for x in number_of_files:
            if x.split('.')[-1] == self.split:
                valid_files.append(f'{self.folder_path}/{str(x)}')
        
        return valid_files

    def _consume_text(self):
        valid_files = self._find_files()
        for x in tqdm(valid_files):
            if x.split('.')[0].split('/')[-1] == "english":
                self.en_data = get_data(x)
            elif x.split('.')[0].split('/')[-1] == "hindi":
                self.hi_data = get_data(x)
    
    def __getitem__(self, idx):
        en_data = self.en_data[idx]
        hi_data = self.hi_data[idx]

        X_toks = self.en_tokenizer(en_data)
        y_toks = self.hi_tokenizer(hi_data)

        X_ids = self.vocab_en.convert_to_seq(X_toks, length=self.max_length)
        y_ids = self.vocab_hi.convert_to_seq(y_toks, length=self.max_length)

        return torch.LongTensor(list(map(int, X_ids))), torch.LongTensor(list(map(int, y_ids)))


# if __name__ == "__main__":
#     vocab_hi = Vocab(lang="hi")
#     vocab_hi.load_vocab(vocab_file="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT/hindi.train_saved_vocab.json")
#     vocab_en = Vocab(lang="en")
#     vocab_en.load_vocab(vocab_file="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT/english.train_saved_vocab.json")

#     english2hindidata = English2HindiData(
#         folder_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT", split="train", vocab_en=vocab_en, vocab_hi=vocab_hi, 
#         max_length=50
#     )

#     dataloader = DataLoader(english2hindidata, batch_size=32)

#     for x in dataloader:
#         print(x[0].shape, x[1].shape)
#         break
    