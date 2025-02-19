import torch
from models.convolutional.utils import ConvBlock


class TestCNNBlock:
    def test_full_pass_mnist(self):
        x = torch.randn(size=(1, 28, 28))
        
        model = ConvBlock(
            type_in="2d",
            in_channels=1,
            out_channels=3,
            kernel_size=2,
            padding=1,
            stride=1,
            act_fn=torch.nn.GELU(),
            pool_=torch.nn.MaxPool2d((2,2))
        )

        y = model.forward(
            x=x,
            verbose=True
        )

        assert y.shape == (3, 14, 14)

    def test_full_pass_cifar(self):
        x = torch.randn(size=(3, 32, 32))
        
        model = ConvBlock(
            type_in="2d",
            in_channels=3,
            out_channels=10,
            kernel_size=2,
            padding=1,
            stride=1,
            act_fn=torch.nn.GELU(),
            pool_=torch.nn.MaxPool2d((2,2))
        )

        y = model.forward(
            x=x,
            verbose=True
        )

        assert y.shape == (10, 16, 16)
    
    def test_1d_operator(self):
        x = torch.randn(size=(3, 32))
        
        model = ConvBlock(
            type_in="1d",
            in_channels=3,
            out_channels=10,
            kernel_size=2,
            padding=1,
            stride=1,
            act_fn=torch.nn.GELU(),
            pool_=torch.nn.MaxPool1d(2)
        )

        y = model.forward(
            x=x,
            verbose=True
        )

        assert y.shape == (10, 16)