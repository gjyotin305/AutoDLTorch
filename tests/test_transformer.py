import torch
from models.transformers.text.NMT.encoder import EncoderLayer
from models.transformers.text.NMT.decoder import DecoderLayer
from models.transformers.utils import PositionalEncoding

class TestUtils:
    def test_pe(self):
        test = torch.rand(64, 30, 512)
        pe = PositionalEncoding(
            d_model=512,
            max_len=128
        )

        position_enc = pe(test)

        assert position_enc.shape == test.shape

class TestComponents:
    def test_encoder(self):
        test = torch.rand(64, 30, 512)
        enc = EncoderLayer(
            d_model=512,
            d_hidden=768,
            n_head=2,
            drop_prob=0.1
        )

        embed = enc(test)

        assert embed.shape == test.shape
    
    # def test_decoder(self):
    #     test = torch.rand(64, 30, 512)
    #     test_2 = torch.rand(64, 30)
    #     dec = DecoderLayer(
    #         d_model=512,
    #         d_hidden=768,
    #         n_head=2,
    #         drop_prob=0.1
    #     )

    #     embed = dec(test)

    #     print(embed.shape)