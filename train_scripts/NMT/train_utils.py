# Standard Imports
import random
import re
import unicodedata
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from constants import MAX_LENGTH, SOS_TOKEN, EOS_TOKEN

## Pre Defined Constants

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS"}
        self.n_words = 2
   
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!= 'Mn')


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, data_folder, reverse=False):
    lines = open(f'{data_folder}{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepare_data(lang1, lang2, data_path, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, data_path, reverse)
    print(f"Sentence Pairs: {len(pairs)}")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)}")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words")
    print("Stats")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

if __name__ == "__main__":
    data_path = "/Users/gjyotin305/Downloads/data/fra-eng/"
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', data_path, True)
    print(random.choice(pairs))
