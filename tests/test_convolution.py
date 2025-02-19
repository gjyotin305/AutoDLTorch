import torch
from models.convolutional.convolution import CNN


class TestCNN:
    def test_full_pass_mnist(self, x):
        x = torch.randn(size=(1, 28, 28))
        