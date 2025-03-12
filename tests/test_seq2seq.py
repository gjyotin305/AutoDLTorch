import torch
from models.seq2seq.sequence_models import (
    ElmanNetwork,
    GatedArch
)

class TestSeqArch:
    def test_elman_rnn(self):
        input = torch.rand(size=(64, 100, 100)) # (batch, timesteps, features)
        model = ElmanNetwork(
            input_dim=100,
            recurrent_dim=256,
            output_dim=10
        )

        out = model.forward(input)
        
        assert out.shape == (64, 10)

    def test_gated_arch(self):
        input = torch.rand(size=(64, 100, 100)) # (batch, timesteps, features)
        model = GatedArch(
            input_dim=100,
            recurrent_dim=256,
            output_dim=10
        )

        out = model.forward(input)
        
        assert out.shape == (64, 10)