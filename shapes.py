import typing
import torch

BT = typing.NewType('BT', torch.Tensor)

T = typing.NewType('T', torch.Tensor)

t = typing.NewType('t', torch.Tensor)

BCt = typing.NewType('BCt', torch.Tensor)
