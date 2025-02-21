import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        type_in,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        act_fn,
        pool_ = None
    ):
        super(ConvBlock, self).__init__()
        self.type_in = type_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.act_fn = act_fn
        self.pool_ = pool_
        self._make_block()

    def _make_block(self):
        if self.type_in == "1d" and self.pool_:
            self.model_comp = [
                nn.Conv1d(
                    in_channels=self.in_channels, 
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                ),
                self.act_fn,
                self.pool_
            ]
        elif self.type_in == "1d":
            self.model_comp = [
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                ),
                self.act_fn
            ]
        elif self.type_in == "2d" and self.pool_:
            self.model_comp = [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                ),
                self.act_fn,
                self.pool_
            ]
        elif self.type_in == "2d":
            self.model_comp = [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                ),
                self.act_fn
            ]
        else:
            raise NotImplementedError

        self.model = nn.Sequential(*self.model_comp)

    def forward(self, x, verbose: bool = False):
        out = self.model(x)
        if verbose:
            print(f"{x.shape} -> {out.shape}")
        return out