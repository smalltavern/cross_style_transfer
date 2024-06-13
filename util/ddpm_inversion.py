import abc
import torch
from torch import inference_mode
from tqdm import tqdm


def invert(x0, pipe, prompt_src="", num_diffusion_steps=100, cfg_scale_src=3.5, eta=1):
    #  inverts a real image according to Algorithm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion
    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps
    pipe.scheduler.set_timesteps(num_diffusion_steps)
    with inference_mode():
        w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()
    wt, zs, wts = inversion_forward_process(pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src,
                                            prog_bar=True, num_inference_steps=num_diffusion_steps)
    return zs, wts

def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0], dim=0)

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                            model_output,
                                            torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def inversion_forward_process(model, x0,
                              etas=None,
                              prog_bar=False,
                              prompt="",
                              cfg_scale=3.5,
                              num_inference_steps=50, eps=None
                              ):
    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps) # 手动加噪过程
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)

    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if not prompt == "":
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts