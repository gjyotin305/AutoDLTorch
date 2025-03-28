import torch
import torch.nn as nn

nn.CrossEntropyLoss()

from models.attention.utils import (
    MultiHeadAttention,
    SelfAttention
)

class TestAttention:
    def test_attention_mha(self):
        self_at = MultiHeadAttention(
            d_model=128,
            num_heads=8
        )
        qkv_proj = nn.Linear(128, 3*128)
        compare_at = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8
        )

        check = torch.rand(64, 100, 128)
        qkv = qkv_proj(check)
        q, k, v = qkv.chunk(3, dim=-1)

        out_self_at = self_at(check)
        out_compare_at = compare_at(q, k, v)

        assert out_compare_at[0].shape == out_self_at.shape

    def test_attention_sa(self):
        self_at = SelfAttention(
            d_model=128
        )
        compare_at = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=1
        )
        
        qkv_proj = nn.Linear(128, 3*128)
        
        check = torch.rand(64, 100, 128)
        qkv = qkv_proj(check)
        q, k, v = qkv.chunk(3, dim=-1)

        out_self_at = self_at(check)
        out_compare_at = compare_at(q, k, v)

        assert out_compare_at[0].shape == out_self_at.shape