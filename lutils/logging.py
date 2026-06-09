from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from lutils.tensor_folder import TensorFolder


@torch.no_grad()
def make_observations_grid(images: torch.Tensor, num_sequences: int) -> torch.Tensor:
    """
    将 [batch_size, channels, height, width] 的图片张量格式化为网格
    """
    # 只保留 num_sequences 批次
    images = images[:num_sequences]

    # 生成网格
    grid = make_grid(images, nrow=num_sequences)
    grid = grid.permute(1, 2, 0)  # 调整维度为 (H, W, C)
    grid = grid.detach().cpu().numpy()  # 转换为 NumPy 数组

    return grid
# def make_observations_grid(
#         images: List[torch.Tensor],
#         num_sequences: int) -> torch.Tensor:
#     """
#     Formats the observations into a grid.

#     :param images: List of [bs, num_observations, 3, height, width]
#     :param num_sequences: Number of sequences to log
#     :return: The grid of observations for logging.
#     """

#     h = max([im.size(3) for im in images])
#     w = max([im.size(4) for im in images])
#     n = max([im.size(1) for im in images])

#     images = [im[:num_sequences] for im in images]

#     def pad(x):
#         if x.size(1) == n:
#             return x
#         else:
#             num_sequences_pad = min(x.size(0), num_sequences)
#             return torch.cat([
#                 torch.zeros([num_sequences_pad, n - x.size(1), 3, h, w]).to(x.device),
#                 x
#             ], dim=1)

#     def resize(x):
#         if x.size(3) == h and x.size(4) == w:
#             return x
#         else:
#             cn = x.size(1)
#             y = F.interpolate(
#                 TensorFolder.flatten(x),
#                 size=(h, w),
#                 mode="nearest")
#             return TensorFolder.fold(y, cn)

#     def add_channels(x):
#         if x.size(2) == 1:
#             return x.expand(-1, -1, 3, -1, -1)
#         else:
#             return x

#     # Pad and resize images
#     images = [to_image(pad(resize(add_channels(x)))) for x in images]

#     # Put the observations one next to another
#     stacked_observations = torch.stack(images, dim=1)
#     flat_observations = TensorFolder.flatten(TensorFolder.flatten(stacked_observations))

#     grid = make_grid(flat_observations, nrow=flat_observations.size(0) // (len(images) * num_sequences))
#     grid = grid.permute(1, 2, 0)
#     grid = grid.detach().cpu().numpy()

#     return grid


@torch.no_grad()
def to_video(x: torch.Tensor) -> np.array:
    return (((torch.clamp(x, -1., 1.) + 1.) / 2.).detach().cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def to_image(x: torch.Tensor) -> torch.Tensor:
    return (((torch.clamp(x, -1., 1.) + 1.) / 2.) * 255).to(torch.uint8)
