import torch as th
import torch.nn as nn


class ReLU1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.minimum(
            th.maximum(th.tensor(0., device=x.device), x),
            th.tensor(1., device=x.device)
        )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.__convs = nn.Sequential(
            nn.ConvTranspose3d(
                32, 20,
                kernel_size=7, stride=2,
                output_padding=1, padding=3
            ),
            nn.SELU(),
            nn.ConvTranspose3d(
                20, 10,
                kernel_size=5, stride=2,
                padding=2, output_padding=1
            ),
            nn.SELU(),
            nn.ConvTranspose3d(
                10, 6,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.SELU(),
            nn.ConvTranspose3d(
                6, 1,
                kernel_size=3, stride=1,
                padding=1
            ),
            nn.Hardsigmoid()
        )

    def forward(self, x_rand: th.Tensor) -> th.Tensor:
        o = self.__convs(x_rand)
        return o


class Disciminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.__convs = nn.Sequential(
            nn.Conv3d(
                1, 4,
                kernel_size=3, padding=1
            ),
            nn.MaxPool3d(2, 2),
            nn.SELU(),
            nn.Conv3d(
                4, 8,
                kernel_size=5, padding=2
            ),
            nn.MaxPool3d(2, 2),
            nn.SELU(),
            nn.Conv3d(
                8, 10,
                kernel_size=5, padding=2
            ),
            nn.MaxPool3d(2, 2),
            nn.SELU()
        )

        self.__lins = nn.Sequential(
            nn.Linear(10 * 8 ** 3, 4096),
            nn.SELU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, data: th.Tensor) -> th.Tensor:
        o = self.__convs(data)
        o = o.flatten(1, -1)
        o = self.__lins(o)
        return o


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake))


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_fake))


if __name__ == '__main__':
    gen = Generator()
    disc = Disciminator()

    x = th.rand(3, 32, 8, 8, 8)

    o = gen(x)

    print(o.size())

    print((o > 0).sum())
    print(o.size())

    o = disc(o)
    print(o.size())
