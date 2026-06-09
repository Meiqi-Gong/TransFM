
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lutils.configuration import Configuration
from model import Model
import torch
from PIL import Image
import torchvision.transforms as transforms
from lutils.loss import WaveletTransform
from diffusers import AutoencoderKL
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTextModel, CLIPTokenizer
from torchvision.transforms import ToPILImage

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import time

device = 'cuda'
# wavelet_dec = WaveletTransform(scale=2, dec=True).cuda()
model_folder_name = 'runs/FlowMatching_VideoFusion_run-wave_recons_vis+hp_inject_grad/'
# model_folder_name = 'runs/FlowMatching_VideoFusion_run-SD/'
vae_id = '/home/whu/HDD_16T/timer/gmq/MultiModal/stable-diffusion-2-1'
vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").cuda()
# VAEmodel = VAEWithAdapter(device=device).to(device)

config = Configuration('configs/fusion.yaml')
t_steps = config['training']['t_steps']
model = Model(config["model"])
# ckpt_path = model_folder_name + 'checkpoints/step_398000_68.6164_2.5123.pth'
ckpt_path = model_folder_name + 'checkpoints/step_398000.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_210000.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_630000_63.9297_1.5597.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_344000_65.7425_2.4508.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_188000_63.1520_2.4151.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_216000_64.2655_2.4085.pth'
# ckpt_path = model_folder_name + 'checkpoints/step_66000_57.7198_2.2951.pth'

loaded_state = torch.load(ckpt_path, map_location="cpu")
state_dict = loaded_state["model"]

if all(k.startswith("module.") for k in state_dict.keys()):
    # 去除 "module." 前缀
    new_state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
else:
    new_state_dict = state_dict
    
model.load_state_dict(new_state_dict, strict=True)
model.cuda()
model.eval()

@torch.no_grad()
def encode(x: torch.Tensor, sample_posterior: bool = True):
    posterior = vae.encode(x)
    if sample_posterior:
        z = posterior.latent_dist.sample()
    else:
        z = posterior.latent_dist.mode()
    # normalize latent code
    z = z * 0.18215
    return z

@torch.no_grad()
def decode(z: torch.Tensor):
    z = 1.0 / 0.18215 * z
    # sample = self.vae.decode(z).sample
    # print(torch.max(sample), torch.min(sample))
    # sample = self.vae.decode(z).sample/2+0.5
    sample = (vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
    return sample

def tensor_to_np(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    np_img = tensor.permute(1, 2, 0).numpy()
    return np_img
       
transform = transforms.Compose([    
     transforms.ToTensor()
    ])
from pathlib import Path
base_dir = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/M3FD/test/Vis/")
root_b = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/M3FD/test/Ir/")
# base_dir = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/FLIR/test/Vis/")
# root_b = Path("/home/whu/HDD_16T/timer/gmq/video/FMFusion_depthfm/FLIR/test/Ir/")
# from glob import glob
# image_paths = glob(os.path.join(base_dir, '*.png'))

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").cuda()
text_encoder = CLIPTextModel.from_pretrained(vae_id, subfolder="text_encoder").cuda()
tokenizer = CLIPTokenizer.from_pretrained(vae_id, subfolder="tokenizer")
wavelet_scale4 = WaveletTransform(scale=3, dec=True).to('cuda')
wavelet_scale5 = WaveletTransform(scale=4, dec=True).to('cuda')
to_pil = ToPILImage()


psnr = 0
ssim = 0
count = 0
total_time = 0
# for img_path in image_paths:
    # file_name = os.path.basename(img_path)    
target_files = sorted([f.name for f in base_dir.glob("*.png")])
# target_files = ["70.png"]
for file_name in target_files:
    
    file_path = os.path.join(base_dir, file_name)
    file_path_b = os.path.join(root_b, file_name)
    
    if os.path.exists(file_path):
        # try:
        img = Image.open(file_path)
        img_b = Image.open(file_path_b)
        img_a = transform(img).cuda().unsqueeze(0)
        img_b = transform(img_b).cuda().unsqueeze(0)
        
        original_size = img_a.shape[-2:]
        target_size = (768, 1024)
        if img_a.shape[-2:] != target_size:
            img_a = F.interpolate(img_a, size=target_size, mode="bilinear", align_corners=False)
            img_b = F.interpolate(img_b, size=target_size, mode="bilinear", align_corners=False)
        
        start_time = time.time()
        blip_inputs = [to_pil(img_a[i]) for i in range(img_a.size(0))]
        inputs = blip_processor(images=blip_inputs, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=77)
        visible_text = blip_processor.decode(out[0], skip_special_tokens=True)
        full_prompt = visible_text + ", transfering to infrared image with thermal targets and rich details"
        inputs = tokenizer(text=full_prompt, return_tensors="pt", padding="max_length", truncation=True).to(device)
        text_embeddings = text_encoder.get_input_embeddings()(inputs["input_ids"])    
        
        img_a = img_a*2.0-1.0                    
        input_latent = encode(img_a)
        inputs_hf = wavelet_scale4(img_a)
        l_high_freqy4 = inputs_hf[:, 3:, :, :]
        inputs_hf = wavelet_scale5(img_a)
        l_high_freqy5 = inputs_hf[:, 3:, :, :]
        image_trans = model.generate_frames(
                    observations=input_latent,
                    conditioning_latents=[l_high_freqy4, l_high_freqy5],
                    context_ca=text_embeddings,
                    steps=2)
        img_out = decode(image_trans)
        inference_time = time.time() - start_time
        
        ir_np = tensor_to_np(img_b[0])
        fake_np = tensor_to_np(img_out[0])
        psnr_output = compare_psnr(ir_np, fake_np, data_range=1.0)
        ssim_output = compare_ssim(ir_np, fake_np, data_range=1.0, channel_axis=-1)
        last_part = file_name.split("/")[-1] 
        print(f"[{last_part}] PSNR: {psnr_output:.4f}, SSIM: {ssim_output:.4f}")
        psnr += psnr_output
        ssim += ssim_output
        count += 1
        total_time += inference_time
        if img_out.shape[-2:] != original_size:
            img_out = F.interpolate(img_out, size=original_size, mode="bilinear", align_corners=False)
            
        img_out = img_out.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
        img_save = (img_out * 255).astype(np.uint8)

        cv2.imwrite('outputs_tmp/'+last_part, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

avg_psnr = psnr / count
avg_ssim = ssim / count
avg_time = total_time / count
print(f"Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}, Avg Time: {avg_time:.2f}s")