import sys
from typing import List
import pyrallis
from AppearanceTransferModel import AppearanceTransferModel
from config import RunConfig, Range
from diffusers.training_utils import set_seed
from util.latent_utils import load_latents_or_invert_images, get_init_latents_and_noises
import torch

@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)
    
def run(cfg: RunConfig):
    # 1、保存所有设置
    # pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    # 2、设置随机种子
    set_seed(cfg.seed)
    # 3、加载SD模型，这只unet结构
    model = AppearanceTransferModel(cfg)
    # 4、DDIM Inversion (content and style Images)
    latents_style, latents_content, zs_style, zs_content = load_latents_or_invert_images(model=model, cfg=cfg)
    # 5、load content and style latent to AppearanceTransferModel
    model.set_latents(latents_style, latents_content)
    model.set_noise(zs_style, zs_content)
    # 6、style transfer to content image
    print("Starting running style transfer.......")
    images = style_transfer(model, cfg)
    print("style transfer is over")
    
def style_transfer(model, cfg):
    # 6.1、get initial latents and noise
    init_latents, init_zs = get_init_latents_and_noises(model, cfg)
    # 6.2、set timesteps and enable edit
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True
    # 6.3、run inference from start step to end step
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    images[0].save(cfg.output_path / f"style_transfer.jpg")
    images[1].save(cfg.output_path / f"style.jpg")
    images[2].save(cfg.output_path / f"content.jpg")
    
    return images
    
    
if __name__ == "__main__":
    main()
    
    
    