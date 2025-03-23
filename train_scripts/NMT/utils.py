# gdown --id 1bEK6RCdnXIqg8JGrJIMvDaAM-baalGwt For dataset

from torch.utils.data import (
    DataLoader, 
    Dataset
)
import pandas as pd
import os
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
    def __init__(self, lang: str, file_path: str):
        self.lang = lang
        self.file_path = file_path
        self.word2index = {
            "<sos>": 0,
            "<eos>": 1,
            "<pad>": -1
        }
        self.index2word = {
            0: "<sos>",
            1: "<eos>",
            -1: "<pad>"
        }
        if lang == "en":
            self.tokenizer = Tokenizer(English().vocab)
        elif lang == "hi":
            self.tokenizer = Tokenizer(Hindi().vocab)
        self._consume()

    def add_word(self, word: str) -> int:
        if word in self.word2index:
            return self.word2index[word]
    
        idx = len(self.word2index)
        self.word2index[word] = idx
        self.index2word[idx] = word

        return idx

    

    def convert_toks_to_id(self, toks: List[str]) -> List[int]:
        return [self.word2index[tok] for tok in toks]

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

        return 200



class English2HindiData(Dataset):
    def __init__(
        self, 
        folder_path: str, 
        split: str, 
        vocab_en: Vocab, 
        vocab_hi: Vocab
    ):
        super().__init__()
        self.folder_path = folder_path
        self.split = split
        self.en_tokenizer = Tokenizer(English().vocab)
        self.hi_tokenizer = Tokenizer(Hindi().vocab)
        self._consume_text()
    
    def _find_files(self):
        number_of_files = os.listdir(self.folder_path)
        
        valid_files = []
        for x in number_of_files:
            if x.split('.')[-1] == self.split:
                valid_files.append(str(x.split('.')[-1]))
        
        return valid_files

    def _consume_text(self):
        valid_files = self._find_files()
        for x in tqdm(valid_files):
            if x.split()[0] == "english":
                self.en_data = get_data(x)
            elif x.split()[0] == "hindi":
                self.hi_data = get_data(x)
    
    def __getitem__(self, idx):
        en_data = self.en_data[idx]
        hi_data = self.hi_data[idx]

        X_toks = self.en_tokenizer(en_data)
        y_toks = self.hi_tokenizer(hi_data)


if __name__ == "__main__":
    sample_vocab = Vocab(lang="en", file_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT/english.test")