import torch

from models.transformers.utils import (
    PositionalEncoding
)

class TestUtils:
    def test_pe(self):
        test = torch.rand(64, 30, 512)
        pe = PositionalEncoding(
            d_model=512,
            max_len=128
        )

        position_enc = pe(test)

        assert position_enc.shape == test.shape
