import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import wandb
from tqdm import tqdm
from einops import rearrange
from torchmetrics import Accuracy
from nltk.translate.bleu_score import sentence_bleu
from utils import Vocab, English2HindiData

device = "cuda" if torch.cuda.is_available() else "cpu"


def translate_sentence(sentence, model, vocab: Vocab, tokenizer, max_len=50):
    model.eval()
    tokens = tokenizer(sentence)
    token_ids = [vocab.word2index["<sos>"]] + [vocab[token] for token in tokens] + [vocab.word2index["<eos>"]]
    src_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(device)

    tgt_ids = [vocab.word2index["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model.forward(src_tensor, tgt_tensor)
        next_token = output.argmax(2)[-1, 0].item()
        tgt_ids.append(next_token)
        if next_token == vocab["<eos>"]:
            break

    translated_tokens = [list(vocab.get_stoi().keys())[idx] for idx in tgt_ids]
    return " ".join(translated_tokens[1:-1])

def eval_model():
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indice
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indice
        pe = pe.unsqueeze(0)  # Add batch dimension

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NMTNet(nn.Module):
    def __init__(self, d_model, en_vocab, hi_vocab, d_ff, max_len=50):
        super(NMTNet, self).__init__()
        self.hi_embed = nn.Embedding(hi_vocab, d_model)
        self.en_embed = nn.Embedding(en_vocab, d_model)

        self.pe = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.3,
            dim_feedforward=d_ff
        )
        self.fcc = nn.Linear(d_model, en_vocab)

    def forward(self, x, y):
        x_t = self.pe(self.en_embed(x))
        y_t = self.pe(self.hi_embed(y))

        tgt_mask = torch.triu(
            torch.ones(y_t.size(0), y_t.size(0)), diagonal=1
        ).bool().to(device)

        out = self.transformer(x_t, y_t, tgt_mask)
        out = self.fcc(out)
        out = rearrange(out, "b s d -> s b d")
        return out


if __name__ == "__main__":
    wandb.init(project="transformer-nmt")

    vocab_hi = Vocab("hi", file_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT/hindi.full")
    vocab_en = Vocab("en", file_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT/english.full")
    dataset = English2HindiData(folder_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT", split="train", max_length=48, vocab_en=vocab_en, vocab_hi=vocab_hi)
    eval_data = English2HindiData(folder_path="/home/gjyotin305/Desktop/AutoDLTorch/train_scripts/NMT/MT", split="test", max_length=48, vocab_en=vocab_en, vocab_hi=vocab_hi)

    test_loader = DataLoader(eval_data, batch_size=32, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NMTNet(
        d_model=128, 
        en_vocab=len(vocab_en.word2index), 
        hi_vocab=len(vocab_hi.word2index), 
        d_ff=512
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    EPOCHS = 20
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    for epoch in tqdm(range(EPOCHS)):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(tqdm(train_loader)):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt)
            # print(output.shape)
            output = rearrange(output, "s b v -> (s b) v")
            tgt_output = rearrange(tgt, "s b -> (s b)")

            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] Completed - Avg Loss: {avg_loss:.4f}"
        )
        wandb.log({
            "loss": avg_loss,
            "epoch": epoch
        })

        if epoch+1 % 5 == 0:
            torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pt")
            print("Model Saved")

    



