import os
from typing import Any, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from evaluation.evaluator import Evaluator
from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from torchvision.utils import make_grid
from lutils.loss import SSIM_loss, L_Gradient, WaveletTransform
from diffusers import AutoencoderKL
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTextModel, CLIPTokenizer
from torchvision.transforms import ToPILImage
# from diffusers import StableDiffusionPipeline

class Trainer:
    """
    Class that handles the training
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: Dataset,
            sampler: None,
            # sampler: torch.utils.data.distributed.Sampler,
            num_gpus: int,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device

        # Create folder for saving
        self.run_path = os.path.join("runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, 'logs'))

        # Setup dataloader
        self.dataset = dataset
        # print(num_gpus)
        self.sampler = (
            sampler if sampler is not None 
            else DistributedSampler(dataset, num_replicas=num_gpus, rank=rank) if num_gpus > 1 
            else RandomSampler(dataset)
            )
        # print(self.sampler)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=False, # if num_gpus > 1 else True,
            num_workers=self.config["batching"]["num_workers"],
            sampler=self.sampler,
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM_loss(window_size=32)
        self.grad = L_Gradient()

        # Optimizer will be defined in train_epoch
        self.optimizer = None

        # Scheduler will be defined in train_epoch
        self.lr_scheduler = None

        self.global_step = 2000
        self.t_steps = self.config['t_steps']
        vae_id = '/home/whu/HDD_16T/timer/gmq/MultiModal/stable-diffusion-2-1'
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(device)
        self.scale_factor = 0.18215
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(vae_id, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(vae_id, subfolder="tokenizer")
        self.wavelet_scale4 = WaveletTransform(scale=3, dec=True).to(device)
        self.wavelet_scale5 = WaveletTransform(scale=4, dec=True).to(device)
        self.to_pil = ToPILImage()

    def init_optimizer(self, model: nn.Module):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(0.9, 0.999))
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config["optimizer"]["num_warmup_steps"],
            num_training_steps=self.config["optimizer"]["num_training_steps"],
            power=0.5)

    def get_lr(self):
        assert self.optimizer is not None

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(
            self,
            model: nn.Module,
            logger: Logger,
            evaluator: Evaluator,
            scalar_logging_frequency: int = 10,
            media_logging_frequency: int = 50,
            saving_frequency: int = 2000,
            evaluation_frequency: int = 2000,
            checkpointing_frequency: int = 4000):
        """
        Trains the model for one epoch

        """
        model.train()
        dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module

        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        # Setup loading bar
        train_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)
        for batch in train_gen:
            if self.num_gpus > 1:
                torch.distributed.barrier()
            observations = batch['gt_vi'].cuda()
            observations_gt = batch['gt_ir'].cuda()

            
            blip_inputs = [self.to_pil((observations[i]+1.0)/2.0) for i in range(observations.size(0))]
            with torch.no_grad():
                self.blip_model.eval()
                inputs = self.blip_processor(images=blip_inputs, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs, max_new_tokens=77)
                visible_texts = [self.blip_processor.decode(o, skip_special_tokens=True) for o in out]
                full_prompt = [text + ", transfering to infrared image with thermal targets and rich details" for text in visible_texts]
                inputs = self.tokenizer(text=full_prompt, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
                text_embeddings = self.text_encoder.get_input_embeddings()(inputs["input_ids"])

            # Forward the model
            inputs = observations
            gt = observations_gt
            
            inputs_hf = self.wavelet_scale4(inputs)
            l_high_freqy4 = inputs_hf[:, 3:, :, :]
            inputs_hf = self.wavelet_scale5(inputs)
            l_high_freqy5 = inputs_hf[:, 3:, :, :]
            input_latent = self.encode(inputs)
            gt_latent = self.encode(gt)
            condition = [l_high_freqy4, l_high_freqy5]
            model_outputs = model(input_latent, gt_latent, condition, text_embeddings)

            # Compute the loss
            loss, auxiliary_output = self.calculate_loss(model_outputs)

            # Backward pass
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.num_gpus > 1:
                torch.distributed.barrier()
            if self.is_main_process:
                if self.global_step % scalar_logging_frequency == 0 and self.is_main_process:
                    self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                    self.writer.add_scalar('Flow_matching_loss/train', auxiliary_output['flow_matching_loss'].item(), self.global_step)
                    self.writer.add_scalar('Reconstructed_loss/train', auxiliary_output['recons_loss'].item(), self.global_step)

                # Log media
                if self.global_step % media_logging_frequency == 0 and self.is_main_process:
                    model_outputs["generated_observations"] = dmodel.generate_frames(
                    observations=input_latent,
                    conditioning_latents=condition,
                    context_ca=text_embeddings,
                    steps=self.t_steps)
                    
                    
                    img_out = self.decode(model_outputs["generated_observations"][0:1])
                    tb_img_first = [
                        (inputs[0] / 2 + 0.5).detach().float().cpu(), 
                        (gt[0] / 2 + 0.5).detach().float().cpu(),
                        img_out[0].detach().float().cpu()
                    ]

                    tb_img = tb_img_first# + tb_img_second
                    tb_img = make_grid(tb_img, nrow=3, padding=2)
                    self.writer.add_image('images_train', tb_img, self.global_step)


                # Evaluate the model
                if self.global_step % evaluation_frequency == 0 and self.is_main_process:
                    psnr, ssim = evaluator.evaluate(model=model, wavelet_scale4=self.wavelet_scale4, wavelet_scale5=self.wavelet_scale5, writer = self.writer, global_step=self.global_step,
                                    vae=self.vae
                                    , blip_processor=self.blip_processor, blip_model=self.blip_model,
                                    tokenizer=self.tokenizer, text_encoder=self.text_encoder)
                    if not hasattr(self, "best_psnr"):
                        self.best_psnr = -float('inf')

                    # 判断是否要保存
                    if psnr > self.best_psnr:
                        self.best_psnr = psnr  # 更新最好psnr
                        state_dict = {
                            "optimizer": self.optimizer.state_dict(),
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "model": model.state_dict(),
                            "global_step": self.global_step
                        }
                        
                        # 保存为 step_{global_step}.pth
                        step_path = os.path.join(self.run_path, "checkpoints", f"step_{self.global_step}_{psnr:.4f}_{ssim:.4f}.pth")
                        torch.save(state_dict, step_path)
                        print(f"[Rank {self.rank}] New best checkpoint saved at {step_path} with PSNR {psnr:.4f} SSIM {ssim:.4f}")

                        # 同时保存一份 best.pth
                        # best_path = os.path.join(self.run_path, "checkpoints", "best.pth")
                        # torch.save(state_dict, best_path)
                        # print(f"[Rank {self.rank}] Also saved as best.pth")
                    else:
                        print(f"[Rank {self.rank}] No improvement, PSNR {psnr:.4f} <= Best PSNR {self.best_psnr:.4f}, not saving.")

            if self.num_gpus > 1:
                torch.distributed.barrier()

            if self.num_gpus > 1:
                torch.distributed.barrier()
            self.global_step += 1

        train_gen.close()

        # Save the model
        logger.info("Saving the trained model...")
        self.save_checkpoint(model, f"final_step_{self.global_step}.pth")
    
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
        # sample = self.vae.decode(z).sample
        # print(torch.max(sample), torch.min(sample))
        # sample = self.vae.decode(z).sample/2+0.5
        sample = (self.vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
        return sample
    
    def calculate_loss(
            self,
            results: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
            # results: DictWrapper[str, Any]) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors)

        # Sum up all the losses
        # loss_weights = self.config["loss_weights"]
        loss = 8 * flow_matching_loss
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
        recons_loss = recons_loss_x1# + 5*recons_loss_refinedx1
        # loss += recons_loss_x1
        loss += recons_loss
        
        # DDP hack
        def add_zero_to_loss(value):
            if v is None:
                return loss
            return loss + value.mul(0).mean()

        for _, v in results.items():
            if isinstance(v, list):
                for ev in v:
                    loss = add_zero_to_loss(ev)
            else:
                loss = add_zero_to_loss(v)

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss,

            # Loss terms
            flow_matching_loss=flow_matching_loss,
            recons_loss = recons_loss
        )

        return loss, auxiliary_output

    def log_scalars(self, loss_terms: Dict[str, Any], logger: Logger):
        for k, v in loss_terms.items():
            logger.log(f"Training/Loss/{k}", v)

        # Log training stats
        logger.log(f"Training/Stats/learning_rate", self.get_lr())
        logger.log(f"Training/Stats/total_loss_is_nan", torch.isnan(loss_terms.total_loss).to(torch.int8))
        logger.log(f"Training/Stats/total_loss_is_inf", torch.isinf(loss_terms.total_loss).to(torch.int8))


    @staticmethod
    # def log_media(results: DictWrapper[str, Any], logger: Logger):
    def log_media(results: Dict[str, Any], logger: Logger):
        num_sequences = min(4, results.observations.size(0))

        obs = results.observations[:num_sequences]  # [num_sequences, 3, H, W]
        gen_obs = results.generated_observations[:num_sequences] 
        grid = make_grid(torch.cat([obs, gen_obs], dim=0), nrow=num_sequences)
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()
        logger.log(f"Training/Media/reconstructed_observations", logger.wandb().Image(grid))

    @staticmethod
    def reduce_gradients(model: nn.Module, num_gpus: int):
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        # if self.num_gpus > 1:
        #     check_ddp_consistency(model, r".*\..+_(mean|var|tracked)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            if checkpoint_name:
                torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])

        ckpt_model_state = loaded_state["model"]
        model_state = model.state_dict()

        new_ckpt_model_state = {}

        # 判断是需要添加 module. 还是去除 module.
        need_prefix = False
        need_remove_prefix = False

        ckpt_keys = list(ckpt_model_state.keys())
        model_keys = list(model_state.keys())

        if all(k.startswith("module.") for k in ckpt_keys) and not all(k.startswith("module.") for k in model_keys):
            need_remove_prefix = True
        elif not all(k.startswith("module.") for k in ckpt_keys) and all(k.startswith("module.") for k in model_keys):
            need_prefix = True

        if need_remove_prefix:
            print("[Checkpoint] Removing 'module.' prefix from loaded checkpoint keys")
            new_ckpt_model_state = {k.replace("module.", "", 1): v for k, v in ckpt_model_state.items()}
        elif need_prefix:
            print("[Checkpoint] Adding 'module.' prefix to loaded checkpoint keys")
            new_ckpt_model_state = {f"module.{k}": v for k, v in ckpt_model_state.items()}
        else:
            new_ckpt_model_state = ckpt_model_state

        # dmodel = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        # dmodel.load_state_dict(state)

        # ckpt_path = model_folder_name + 'checkpoints/final_step_300000.pth'
        # loaded_state = torch.load(ckpt_path, map_location="cpu")
        # dmodel.load_state_dict(loaded_state["model"], strict=True)
        # model.load_state_dict(new_ckpt_model_state, strict=True)
        ckpt_model_state = new_ckpt_model_state if isinstance(model, torch.nn.parallel.DistributedDataParallel) else ckpt_model_state
        model.load_state_dict(new_ckpt_model_state, strict=True)

        self.global_step = loaded_state["global_step"]
