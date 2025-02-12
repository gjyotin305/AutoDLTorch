import torch
from models.autoencoders.autoencoder import (
    AutoEncoder,
    ConvAutoEncoder,
    SimpleAutoEncoder
) 


class TestSimpleAutoEncoder:
    def test_full_pass(self):
        input_test = torch.rand((64, 28*28))
        model = SimpleAutoEncoder(
            input_dim=28*28,
            hidden_dim=14*14,
            n_layers=2
        )

        output_test = model.forward(input_test)

        assert output_test.shape == input_test.shape


class TestAutoEncoders:
    def test_encoder_forward(self):
        input_test = torch.rand(size=(64, 28*28))
        
        model = AutoEncoder(
            input_dim=28*28,
            hidden_dim=14*14,
            activation_fn=torch.nn.ReLU(),
            n_layers=3
        )
        
        output_test = model.encoder_forward(input_test)

        assert output_test.shape[1] < input_test.shape[1]
        
    def test_full_pass(self):
        input_test = torch.rand(size=(64, 28*28))
        
        model = AutoEncoder(
            input_dim=28*28,
            hidden_dim=14*14,
            activation_fn=torch.nn.ReLU(),
            n_layers=3
        )
        
        output_test = model.forward(input_test)

        assert output_test.shape == input_test.shape