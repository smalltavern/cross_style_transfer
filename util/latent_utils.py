from AppearanceTransferModel import AppearanceTransferModel
from config import RunConfig
from util.image_utils import load_images
import torch
import numpy as np
from util.ddpm_inversion import invert



def load_latents_or_invert_images(model, cfg):
    print("DDIM Inversion Start(content and style).......")
    image_style, image_content = load_images(cfg)
    model.enable_edit = False
    latents_style, latents_content, noise_style, noise_content = invert_images(sd_model=model.pipe, 
                                                                         style_image=image_style, 
                                                                         content_image=image_content, 
                                                                         cfg=cfg)
    model.enable_edit = True
    print("DDIM Inversion is over...")
    return latents_style, latents_content, noise_style, noise_content
    
    

def invert_images(sd_model, style_image, content_image, cfg):
    input_style = torch.from_numpy(np.array(style_image)).float() / 127.5 - 1.0
    input_content = torch.from_numpy(np.array(content_image)).float() / 127.5 - 1.0
    zs_style, latents_style = invert(x0=input_style.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
    zs_content, latents_content = invert(x0=input_content.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    
    return latents_style, latents_content, zs_style, zs_content


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig):
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_content.dim() == 4 and model.latents_style.dim() == 4 and model.latents_style.shape[0] > 1:
        model.latents_content = model.latents_content[cfg.skip_steps]
        model.latents_style = model.latents_style[cfg.skip_steps]
    init_latents = torch.stack([model.latents_content, model.latents_style, model.latents_content])
    init_zs = [model.zs_content[cfg.skip_steps:], model.zs_style[cfg.skip_steps:], model.zs_content[cfg.skip_steps:]]
    return init_latents, init_zs
    
    
    
    