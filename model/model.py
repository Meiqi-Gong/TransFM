from typing import Dict, Any

import torch
import torch.nn as nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.SD_Unet_Hp import VectorFieldRegressor

class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]
        self.noising_step = config['noising_step']
        self.t_steps = config['t_steps']
        self.register_buffer('flow_scaling', torch.linspace(0.0, 1.0, self.t_steps))

        ckpt_path = "/home/whu/HDD_16T/timer/gmq/MultiModal/depth-fm-main/checkpoints/depthfm-v1.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # self.vector_field_regressor = build_vector_field_regressor(
        #     config=self.config["vector_field_regressor"], ckpt="/home/whu/HDD_16T/timer/gmq/MultiModal/stable-diffusion-v1-5/v1-5-pruned.ckpt")
        # ckpt['ldm_hparams']['in_channels'] = 4+189+4
        # ckpt['ldm_hparams']['in_channels'] = 4
        self.vector_field_regressor = VectorFieldRegressor(**ckpt['ldm_hparams'])
        
    
    
    def q_sample(self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
        """
        Diffuse the data for a given number of diffusion steps. In other
        words sample from q(x_t | x_0).
        """
        def cosine_log_snr(t, eps=0.00001):
            """
            Returns log Signal-to-Noise ratio for time step t and image size 64
            eps: avoid division by zero
            """
            return -2 * np.log(np.tan((np.pi * t) / 2) + eps)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def cosine_alpha_bar(t):
            return sigmoid(cosine_log_snr(t))
        dev = x_start.device
        dtype = x_start.dtype

        if noise is None:
            noise = torch.randn_like(x_start)
        
        alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
        alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)

        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise

    def predict_start_from_vector(self, x_t, t, vector):
        t_index = t * (self.t_steps - 1)  # 将 t∈[0,1] 映射到索引范围 [0, num_steps-1]
        t_floor = t_index.floor().long()    # 向下取整
        t_ceil = t_floor + 1                # 向上取整
        t_frac = t_index - t_floor          # 小数部分

        # 线性插值
        scaling = (1 - t_frac) * self.flow_scaling[t_floor] + t_frac * self.flow_scaling[t_ceil]
        return x_t + scaling * vector
    
    def forward(
            self,
            observations: torch.Tensor,
            observations_gt: torch.Tensor,
            conditioning_latents: torch.Tensor,
            context_ca: torch.Tensor,
            ) -> Dict[str, Any]:
            # observations: torch.Tensor) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """
        # self.load_from_depthfm()
        batch_size = observations.size(0)
        # noise = torch.randn_like(observations_gt).to(observations_gt.dtype).to(observations_gt.device)
        t = torch.rand(batch_size, 1, 1, 1).to(observations_gt.dtype).to(observations_gt.device)
        x_0 = self.q_sample(observations, self.noising_step)
        x_1 = observations_gt
        sigma = 0.0000001
        x_t = (1 - (1 - sigma) * t) * x_0 + t * x_1
        target_vectors = (x_1 - (1 - sigma) * x_t) / (1 - (1 - sigma) * t)
        # context_ca = torch.tensor(self.empty_text_embed).cuda().repeat(batch_size, 1, 1)
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=x_t,
            conditioning_latents=[observations, conditioning_latents],
            timestamps=t.squeeze(3).squeeze(2).squeeze(1),
            context_ca = context_ca)
        x_recon = reconstructed_vectors * (1 - (1 - sigma) * t) + (1 - sigma) * x_t

        return DictWrapper(
            # Inputs
            target=observations_gt,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            reconstructed_x1=x_recon,
            # reconstructed_x1=[denoise_x0, x_recon],
            target_vectors=target_vectors)

    def reconstruct_x1(
            self,
            observations: torch.Tensor,
            conditioning_latents: torch.Tensor,
            context_ca: torch.Tensor,
            steps: int = 4,
        ) -> torch.Tensor:

        def f(t: torch.Tensor, y: torch.Tensor):
            # 1. 预测向量场
            vectors = self.vector_field_regressor(
                input_latents=y,
                conditioning_latents=[observations, conditioning_latents],
                timestamps=t * torch.ones(b).to(y.device),
                context_ca = context_ca)

            return vectors  # 直接返回向量场，不做额外处理

        b, c, h, w = observations.shape
        y0 = self.q_sample(observations, self.noising_step)
        
        # ODE 积分（从加噪观测 y0 到生成结果 y1）
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0/steps))
        t = torch.linspace(0, 1, steps+1).to(y0.device)  # 包含初始点
        ode_results = odeint(f, y0, t, **ode_kwargs)

        # # # 取最终解 y_T，并进行 refinement
        # y_T = ode_results[-1]  # ODE 过程结束后的 y_T 直接是图像
        # refined_x = self.refinement_model(
        #     input_latents=y_T,
        #     conditioning_latents=conditioning_latents,
        #     timestamps=torch.tensor(1.0, device=y_T.device).view(1).repeat(b))

        return ode_results[-1]  # 返回最终优化后的图像
        
    @torch.no_grad()
    def generate_frames(
            self,
            observations: torch.Tensor,
            conditioning_latents: torch.Tensor,
            context_ca: torch.Tensor,
            steps: int = 4,
        ) -> torch.Tensor:

        def f(t: torch.Tensor, y: torch.Tensor):
            # 1. 预测向量场
            vectors = self.vector_field_regressor(
                input_latents=y,
                # conditioning_latents=torch.cat([observations, conditioning_latents],1),
                # conditioning_latents=conditioning_latents,
                conditioning_latents=[observations, conditioning_latents],
                timestamps=t * torch.ones(b).to(y.device),
                context_ca = context_ca)

            return vectors  # 直接返回向量场，不做额外处理

        b, c, h, w = observations.shape
        y0 = self.q_sample(observations, self.noising_step)
        # y0 = observations
        # context_ca = torch.tensor(self.empty_text_embed).cuda().repeat(b, 1, 1)
        # context_ca = self.empty_text_embed.cuda().repeat(b, 1, 1)
        
        # ODE 积分（从加噪观测 y0 到生成结果 y1）
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0/steps))
        t = torch.linspace(0, 1, steps+1).to(y0.device)  # 包含初始点
        ode_results = odeint(f, y0, t, **ode_kwargs)

        # # 取最终解 y_T，并进行 refinement
        y_T = ode_results[-1]  # ODE 过程结束后的 y_T 直接是图像
        # refined_x = self.refinement_model(
        #     input_latents=y_T,
        #     conditioning_latents=torch.cat([observations, conditioning_latents[0]], 1),
        #     timestamps=torch.tensor(1.0, device=y_T.device).view(1).repeat(b))

        return y_T  # 返回最终优化后的图像
    


import pickle
import math
class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='/home/whu/HDD_16T/timer/gmq/MultiModal/WaveDM-main/models/wavelet_weights_c2.pkl',
                 transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                dct = u.load()
                # dct = pickle.load(f)
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            # pdb.set_trace()
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output
