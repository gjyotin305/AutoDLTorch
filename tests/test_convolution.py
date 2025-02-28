import torch
from models.convolutional.utils import ConvBlock
from models.convolutional.convolution import CNN2DClassification
from models.convolutional.resnet import BasicBlock


class TestResNetBlock:
    def test_full_pass(self):
        x = torch.rand(size=(20, 3, 224, 224))
        
        in_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            padding=1
        )

        out = in_conv(x)

        model = BasicBlock(
            in_channel=16,
            kernel_size=3,
            padding=1
        )

        out = model.forward(out)

        print(out.shape)

        assert out.shape == (20, 3, 224, 224)


class TestCNNBlock:
    def test_full_pass_mnist(self):
        x = torch.randn(size=(4, 1, 28, 28))
        
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

        assert y.shape == (4, 3, 14, 14)

    def test_full_pass_cifar(self):
        x = torch.randn(size=(4, 3, 32, 32))
        
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

        assert y.shape == (4, 10, 16, 16)
    
    def test_1d_operator(self):
        x = torch.randn(size=(4, 3, 32))
        
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

        assert y.shape == (4, 10, 16)

class TestCNN:
    def test_conv_pass_cifar10(self):
        x = torch.randn(4, 3, 32, 32)

        model = CNN2DClassification(
            in_channels=3,
            growth_rate=10,
            kernel_size=3,
            padding=1,
            n_layers_conv=3,
            hidden_dim=256,
            pool_kernel=torch.nn.MaxPool2d((2,2)),
            n_classes=10,
        )

        y = model.forward(
            x
        )
        
        assert y.shape == (4, 10)

