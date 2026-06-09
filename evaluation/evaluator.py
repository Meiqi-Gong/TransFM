from typing import Dict, Any
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pathlib
import numpy as np
from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from lutils.running_average import RunningMean
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from lutils.loss import SSIM_loss, L_Gradient
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from torchvision.transforms import ToPILImage
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# from diffusers import AutoencoderKL

class Evaluator:
    """
    Class that handles the evaluation
    """

    def __init__(
            self,
            rank: int,
            config: Configuration,
            dataset: Dataset,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Evaluator, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.device = device

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=True,
            num_workers=self.config["batching"]["num_workers"],
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM_loss()
        self.grad = L_Gradient()

        self.running_means = RunningMean()
        self.t_steps = self.config['t_steps']
        # vae_id = "runwayml/stable-diffusion-v1-5"
        # vae_id = "/home/whu/HDD_16T/timer/gmq/MultiModal/stable-diffusion-2-1"
        # self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(device)
        self.scale_factor = 0.18215

    @torch.no_grad()
    def save_image(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[str, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save a given Tensor into an image file.

        Args:
            tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
                saves the tensor as a grid of images by calling ``make_grid``.
            fp (string or file object): A filename or a file object
            format(Optional):  If omitted, the format to use is determined from the filename extension.
                If a file object was used instead of a filename, this parameter should always be used.
            **kwargs: Other arguments are documented in ``make_grid``.
        """
        grid = make_grid(tensor, **kwargs)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(fp, format=format)
    
    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            # newVAE: nn.Module,
            # wavelet_scale1:nn.Module,
            # wavelet_scale2:nn.Module,
            wavelet_scale4:nn.Module,
            wavelet_scale5:nn.Module,
            writer: SummaryWriter,
            global_step: int,
            vae: nn.Module, 
            blip_processor: nn.Module, blip_model: nn.Module, 
            tokenizer: nn.Module, text_encoder: nn.Module, 
            ):
        """
        Evaluates the model

        """
        # print("Evaluate called in rank:", torch.distributed.get_rank() if torch.distributed.is_initialized() else "not initialized")
        # print("Model type:", type(model))
        # print("isinstance DDP:", isinstance(model, torch.nn.parallel.DistributedDataParallel))
        self.vae=vae
        model.eval()
        transform = transforms.Compose([    
            transforms.ToTensor()
            ])
        to_pil = ToPILImage()

        eval_step = 1
        tb_images = []
        fmeval_loss = 0
        reeval_loss = 0
        psnr = 0
        ssim = 0
        
        base_dir = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/M3FD/test/Vis/")
        root_b = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/M3FD/test/Ir/")
        target_files = ["10.png", "100.png", "150.png"]
        # target_files = ["100.png"]
        for file_path in base_dir.glob("*"):
            if file_path.name in target_files:
                img_vis = Image.open(str(file_path))
                img_ir = Image.open(str(root_b / file_path.name))
                img_a = transform(img_vis).cuda().unsqueeze(0)
                img_b = transform(img_ir).cuda().unsqueeze(0)  
                
                blip_inputs = [to_pil(img_a[i]) for i in range(img_a.size(0))]
                inputs = blip_processor(images=blip_inputs, return_tensors="pt").to(self.device)
                out = blip_model.generate(**inputs, max_new_tokens=77)
                visible_text = blip_processor.decode(out[0], skip_special_tokens=True)
                full_prompt = visible_text + ", transfering to infrared image with thermal targets and rich details"
                inputs = tokenizer(text=full_prompt, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
                text_embeddings = text_encoder.get_input_embeddings()(inputs["input_ids"])    
                
                img_a = img_a*2.0-1.0      
                img_b = img_b*2.0-1.0                      
                input_latent = self.encode(img_a)
                gt_latent = self.encode(img_b)
                # input_latent = newVAE.encode(img_a)
                # gt_latent = newVAE.encode(img_b)
                inputs_hf = wavelet_scale4(img_a)
                l_high_freqy4 = inputs_hf[:, 3:, :, :]
                # B, _, H, W = inputs_hf.shape
                # # l_high_freqy = inputs_hf[:, [i for i in range(192) if i % 64 != 0], :, :]
                # l_high_freqy4 = inputs_hf.view(B, 3, 64, H, W)[:, :, 1:, :, :].reshape(B, -1, H, W)
                inputs_hf = wavelet_scale5(img_a)
                l_high_freqy5 = inputs_hf[:, 3:, :, :]
                # B, _, H, W = inputs_hf.shape
                # # l_high_freqy = inputs_hf[:, [i for i in range(192) if i % 64 != 0], :, :]
                # l_high_freqy5 = inputs_hf.view(B, 3, 256, H, W)[:, :, 1:, :, :].reshape(B, -1, H, W)
                
                # model_outputs = model(input_latent,gt_latent,input_latent)
                model_outputs = model(input_latent, gt_latent, [l_high_freqy4, l_high_freqy5], text_embeddings)
                fm_loss_output, re_loss_output, dict_outs = self.calculate_loss(model_outputs)
                fmeval_loss += fm_loss_output
                reeval_loss += re_loss_output

                # Accumulate scalars
                self.running_means.update(dict_outs)
                # writer.add_scalar('Loss/eval', loss_output.item(), global_step)

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    image_trans = model.module.generate_frames(
                        observations=input_latent,
                        conditioning_latents=[l_high_freqy4, l_high_freqy5],
                        context_ca=text_embeddings,
                        steps=self.t_steps)
                else:
                    image_trans = model.generate_frames(
                        observations=input_latent,
                        conditioning_latents=[l_high_freqy4, l_high_freqy5],
                        context_ca=text_embeddings,
                        steps=self.t_steps)
                
                # x = img_a[0:1]
                # wavelet_out = wavelet_scale4(x)
                # l_high_freq = wavelet_out[:, [i for i in range(192) if i % 64 != 0], :, :]
                # l_high_freq = (l_high_freq - l_high_freq.min()) / (l_high_freq.max() - l_high_freq.min())
                # l_high_freq = l_high_freq*2.0-1.0
                # wavelet_out = wavelet_scale2(x)
                # m_high_freq = wavelet_out[:, [i for i in range(48) if i % 16 != 0], :, :]
                # m_high_freq = (m_high_freq - m_high_freq.min()) / (m_high_freq.max() - m_high_freq.min())
                # m_high_freq = m_high_freq*2.0-1.0
                # wavelet_out = wavelet_scale1(x)
                # h_high_freq = wavelet_out[:, [i for i in range(12) if i % 4 != 0], :, :]
                # h_high_freq = (h_high_freq - h_high_freq.min()) / (h_high_freq.max() - h_high_freq.min())
                # h_high_freq = h_high_freq*2.0-1.0
                # org_high_freq = extract_high_frequency_details(x)
                # org_high_freq = (org_high_freq - org_high_freq.min()) / (org_high_freq.max() - org_high_freq.min())
                # org_high_freq = org_high_freq*2.0-1.0 
                # img_out = self.newVAE(model_outputs["generated_observations"][0:1], l_high_freq, m_high_freq, h_high_freq, org_high_freq)

                img_out = self.decode(image_trans)
                tb_img_first = [
                    (img_a[0] / 2 + 0.5).detach().float().cpu(), 
                    (img_b[0] / 2 + 0.5).detach().float().cpu(),
                    img_out[0].detach().float().cpu()
                ]
                tb_img = tb_img_first
                tb_img = make_grid(tb_img, nrow=3, padding=2)
                tb_images.append(tb_img)
                
                ir_np = self.tensor_to_np(img_b[0] / 2 + 0.5)
                fake_np = self.tensor_to_np(img_out[0])
                psnr_output = compare_psnr(ir_np, fake_np, data_range=1.0)
                ssim_output = compare_ssim(ir_np, fake_np, data_range=1.0, channel_axis=-1)
                psnr += psnr_output
                ssim += ssim_output
                # img_out = img_out.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
                # img_save = (img_out * 255).astype(np.uint8)
                # cv2.imwrite('outputs/100.png', cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
                
                # writer.add_image('images_evaluate', tb_img, global_step)
                # writer.add_image('images_evaluate', tb_img, global_step+eval_step)
                # eval_step += 1
        fmeval_loss = fmeval_loss/eval_step
        reeval_loss = reeval_loss/eval_step
        avg_psnr = psnr / eval_step
        avg_ssim = ssim / eval_step
        writer.add_scalar('Loss/eval_flow_matching', fmeval_loss.item(), global_step)
        writer.add_scalar('Loss/eval_reconstruct', reeval_loss.item(), global_step)
        writer.add_scalar('Loss/eval_PSNR', avg_psnr, global_step)
        writer.add_scalar('Loss/eval_SSIM', avg_ssim, global_step)
        eval_grid = make_grid(tb_images, nrow=1, padding=5)
        writer.add_image("images_evaluate", eval_grid, global_step=global_step)
                
        # for root, dirs, _ in os.walk(base_dir):
        #     for dir_name in dirs:
        #         dir_path = os.path.join(root, dir_name)
        #         dir_b_path = os.path.join(root_b, dir_name)
        #         # target_files = ["1.jpg", "15.jpg", "30.jpg"]
        #         target_files = ["15.jpg"]
        #         for file_name in target_files:
        #             file_path = os.path.join(dir_path, file_name)
        #             file_path_b = os.path.join(dir_b_path, file_name)
                    
        #             if os.path.exists(file_path):
        #                 img = Image.open(file_path)
        #                 img_b = Image.open(file_path_b)
        #                 img_a = transform(img).cuda().unsqueeze(0)
        #                 img_b = transform(img_b).cuda().unsqueeze(0)
        #                 img_a = Wav_Dec(img_a[:,:,:448])
        #                 img_b = Wav_Dec(img_b[:,:,:448])
        #                 model_outputs = model(img_a[:,:3],img_b[:,:3],img_a[:,:3])
        #                 fm_loss_output, re_loss_output, dict_outs = self.calculate_loss(model_outputs)
        #                 fmeval_loss += fm_loss_output
        #                 reeval_loss += re_loss_output

        #                 # Accumulate scalars
        #                 self.running_means.update(dict_outs)
        #                 # writer.add_scalar('Loss/eval', loss_output.item(), global_step)
                        
        #                 image_trans = model.generate_frames(
        #                     observations=img_a[:,:3],
        #                     conditioning_latents=img_a[:,:3],
        #                     steps=self.t_steps,
        #                     warm_start=0.1)
        #                 tb_img_first = [
        #                     img_a[0,:3].detach().float().cpu(), 
        #                     img_b[0,:3].detach().float().cpu(),
        #                     image_trans[-1,0].detach().float().cpu()
        #                 ]
        #                 tb_img = tb_img_first
        #                 tb_img = make_grid(tb_img, nrow=3, padding=2)
        #                 # writer.add_image('images_evaluate', tb_img, global_step)
        #                 writer.add_image('images_evaluate', tb_img, global_step+eval_step)
        #                 eval_step += 1
        # fmeval_loss = fmeval_loss/eval_step
        # reeval_loss = reeval_loss/eval_step
        # writer.add_scalar('Loss/eval_flow_matching', fmeval_loss.item(), global_step)
        # writer.add_scalar('Loss/eval_reconstruct', reeval_loss.item(), global_step)
        # if not self.is_main_process:
        #     return

        # if global_step == 0:
        #     max_num_batches = 10

        # model.eval()

        # # Setup loading bar
        # eval_gen = tqdm(
        #     self.dataloader,
        #     total=min(max_num_batches, len(self.dataloader)),
        #     desc="Evaluation: Batches",
        #     disable=not self.is_main_process,
        #     leave=False)
        # for i, batch in enumerate(eval_gen):
        #     if i >= max_num_batches:
        #         break

        #     # Fetch data
        #     # observations = batch.cuda()
        #     # num_observations = self.config["num_observations"]
        #     # observations = observations[:, :num_observations]
        #     observations = batch['gt_ir'].cuda()
        #     observations_gt = batch['gt_vi'].cuda()
        #     model_outputs = model(observations[:,0], observations_gt[:,0], observations[:,0])

        #     # Forward the model
        #     # model_outputs = model(
        #     #     observations)

        #     # Compute the loss
        #     loss_output = self.calculate_loss(model_outputs)

        #     # Accumulate scalars
        #     self.running_means.update(loss_output)

        #     # Log data only for the 1st batch
        #     if i != 0:
        #         continue

        #     # Log media
        #     dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module
        #     model_outputs["generated_observations"] = dmodel.generate_frames(
        #         observations=observations[:,0],
        #         observations_gt=observations_gt[:,0],
        #         conditioning_latents=observations[:,0],
        #         steps=5,
        #         warm_start=0.1)
        #     self.log_media(model_outputs, logger)

        # # Log scalars
        # for k, v in self.running_means.get_values().items():
        #     logger.log(f"Validation/Loss/{k}", v)

        # # Finalize logs
        # logger.finalize_logs(step=global_step)

        # Close loading bar
        # eval_gen.close()

        # Reset the model to train
        model.train()
        return avg_psnr, avg_ssim

    def tensor_to_np(self, tensor):
        tensor = tensor.detach().cpu().clamp(0, 1)
        np_img = tensor.permute(1, 2, 0).numpy()
        return np_img
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.vae.encode(x)
        if sample_posterior:
            z = posterior.latent_dist.sample()
        else:
            z = posterior.latent_dist.mode()
        # normalize latent code
        z = z * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        z = 1.0 / self.scale_factor * z
        # sample = nn.functional.tanh(self.vae.decode(z).sample)/2+0.5
        # sample = self.vae.decode(z).sample
        # print(torch.max(sample), torch.min(sample))
        sample = (self.vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
        return sample
    
    @torch.no_grad()
    def calculate_loss(
            self,
            results: Dict[str, Any]) -> Dict[str, Any]:
            # results: DictWrapper[str, Any]) -> DictWrapper[str, Any]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors)
        # recons_loss = 4*self.l1_loss(
        #     results.reconstructed_x1,
        #     results.target
        # ) + self.ssim(
        #     results.reconstructed_x1,
        #     results.target) + 5*self.grad(
        #     results.reconstructed_x1,
        #     results.target)
        # x1, refined_x1 = results.reconstructed_x1
        x1 = results.reconstructed_x1
        recons_loss_x1 = 4*self.l1_loss(
            x1,
            results.target
        ) + self.ssim(
            x1,
            results.target) + 5*self.grad(
            x1,
            results.target)
        # recons_loss_refinedx1 = 4*self.l1_loss(
        #     refined_x1,
        #     results.target
        # ) + self.ssim(
        #     refined_x1,
        #     results.target) + 5*self.grad(
        #     refined_x1,
        #     results.target)
        recons_loss = recons_loss_x1# + 5*recons_loss_refinedx1
        # target = self.tensor_to_np(results.target)
        # psnr = compare_psnr((), results.reconstructed_x1, data_range=1.0)
        # Create auxiliary output
        output = DictWrapper(
            # Loss terms
            flow_matching_loss=flow_matching_loss,
            recons_loss = recons_loss,
            # psnr = psnr
        )

        return flow_matching_loss, recons_loss, output

