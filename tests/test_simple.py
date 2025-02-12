import torch
from models.base.simple import (
    ANN,
    DNN
)

class TestANN:
    def test_forward_pass(self):
        input_test = torch.rand((64, 28*28))
        model = ANN(
            in_dim=28*28,
            hidden_dim=14*14,
            out_dim=10,
            act_fn=torch.nn.ReLU()
        )
        
        output_test = model.forward(input_test)

        assert output_test.shape == (64, 10)

class TestDNN:
    def test_forward_pass(self):
        input_test = torch.rand((64, 28*28))
        model = DNN(
            in_dim=28*28,
            hid_dim=14*14,
            out_dim=10,
            n_layers=2,
            act_fn=torch.nn.ReLU()
        )

        output_test = model.forward(input_test)

        assert output_test.shape == (64, 10)
