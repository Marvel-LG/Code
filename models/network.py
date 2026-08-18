import math
import random
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from models.respace import space_timesteps
import torch.nn as nn
import functools
from torch.autograd import Variable
import torch.nn.functional as F


class Network(BaseNetwork):  # 包含了Unet和时间调度器
    def __init__(self, model_parameter, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        # self.Resampler = None
        if self.pattern == 'A':
            # from models.BOPL_VAE.unet import GlobalGenerator_DCDCv2 as UNet
            from models.BOPL_VAE.unet_org import GlobalGenerator_DCDCv2 as UNet
            # from models.VAE.encode_unet.unet import UNet as UNet
            # from models.VAE.autoencoder.autoencoder import AautoencoderKL as UNet
            # from models.VAE.encode_a_model.unet import UNet as UNet
            from .discriminator.networks import MultiscaleDiscriminator
            from .discriminator.networks import GANLoss, VGGLoss_torch
            # from .SupCon.resnet_big import SupConResNet
            from .SupCon.PCA import VGG
            from .SupCon.losses import SupConLoss
            from .ContraGan.contraltive_gan import Discriminator
            from .ContraGan.losses import ConditionalContrastiveLoss
            self.denoise_fn = UNet(**model_parameter['unet_A'])
            # self.SupCon = SupConResNet(img_size=64, d_conv_dim=128, apply_d_sn=True, apply_attn=False, attn_d_loc=[],
            #                            d_embed_dim=128 * 3, normalize_d_embed=True, d_init='ortho',
            #                            mixed_precision=False)
            embed_channel = 128
            emned_mutual = 4
            self.SupCon = VGG(in_channle=embed_channel, num_components=embed_channel * emned_mutual)
            self.discriminator_feat = MultiscaleDiscriminator(embed_channel, 64, 3,
                                                              functools.partial(nn.InstanceNorm2d, affine=False), False,
                                                              1,
                                                              True)
            self.discriminator_syn = MultiscaleDiscriminator(3, 64, 3,
                                                             functools.partial(nn.InstanceNorm2d, affine=False),
                                                             False, 2,
                                                             True)
            self.discriminator_y_0 = MultiscaleDiscriminator(3, 64, 3,
                                                             functools.partial(nn.InstanceNorm2d, affine=False),
                                                             False, 2,
                                                             True)
            self.discriminator_contra_gan = None  # Discriminator(img_size=256, d_conv_dim=96, apply_d_sn=True, apply_attn=True,
            # attn_d_loc=[1],
            # d_embed_dim=embed_channel, normalize_d_embed=True,
            # num_classes=3,
            # d_init='ortho',
            # mixed_precision=False)
            self.SupConLoss = SupConLoss(num_classes=3, temperature=1.0)
            self.criterionGAN = GANLoss()  # 判别gan
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss_torch()
            self.criterionConditionalContrastiveLoss = ConditionalContrastiveLoss(num_classes=3, temperature=1.0)
            self.n_layers_D = 3
            self.num_D = 2
        elif self.pattern == 'AandMAP':
            # from models.BOPL_VAE.unet import GlobalGenerator_DCDCv2 as UNet
            # from models.VAE.encode_unet.unet import UNet as UNet
            # from models.VAE.autoencoder.autoencoder import AautoencoderKL as UNet
            from models.VAE.encode_a_model.unet import UNet as UNet
            from .discriminator.networks import MultiscaleDiscriminator
            from .discriminator.networks import GANLoss, VGGLoss_torch
            # from .SupCon.resnet_big import SupConResNet
            from .SupCon.PCA import VGG
            from .SupCon.losses import SupConLoss
            from .ContraGan.contraltive_gan import Discriminator
            from .ContraGan.losses import ConditionalContrastiveLoss
            from .mapping_modules.Mapping import Map
            self.denoise_fn = Map(module_name, model_parameter['unet_A'],
                                  model_parameter['mapping'])
            # self.SupCon = SupConResNet(img_size=64, d_conv_dim=128, apply_d_sn=True, apply_attn=False, attn_d_loc=[],
            #                            d_embed_dim=128 * 3, normalize_d_embed=True, d_init='ortho',
            #                            mixed_precision=False)
            embed_channel = 256
            emned_mutual = 2
            self.SupCon = VGG(in_channle=embed_channel, num_components=embed_channel * emned_mutual)
            self.discriminator_feat = MultiscaleDiscriminator(embed_channel, 64, 3,
                                                              functools.partial(nn.InstanceNorm2d, affine=False), False,
                                                              1,
                                                              True)
            self.discriminator_syn = MultiscaleDiscriminator(3, 64, 3,
                                                             functools.partial(nn.InstanceNorm2d, affine=False),
                                                             False, 2,
                                                             True)
            self.discriminator_y_0 = None  # MultiscaleDiscriminator(3, 64, 3,
            # functools.partial(nn.InstanceNorm2d, affine=False),
            # False, 2,
            # True)
            self.discriminator_contra_gan = None  # Discriminator(img_size=256, d_conv_dim=96, apply_d_sn=True, apply_attn=True,
            # attn_d_loc=[1],
            # d_embed_dim=embed_channel, normalize_d_embed=True,
            # num_classes=3,
            # d_init='ortho',
            # mixed_precision=False)
            self.SupConLoss = SupConLoss(num_classes=3, temperature=1.0)
            self.criterionGAN = GANLoss()  # 判别gan
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss_torch()
            self.criterionConditionalContrastiveLoss = ConditionalContrastiveLoss(num_classes=3, temperature=1.0)
            self.n_layers_D = 3
            self.num_D = 2
        elif self.pattern == 'B':
            from models.discriminator.networks import MultiscaleDiscriminator
            from models.VAE.encode_a_model.unet import UNet as UNet
            from .discriminator.networks import GANLoss, VGGLoss_torch
            self.denoise_fn = UNet(**model_parameter['unet_B'])
            self.discriminator_y_0 = None  # MultiscaleDiscriminator(3, 64, 3,
            # functools.partial(nn.InstanceNorm2d, affine=False),
            # False, 2,
            # True)
            self.criterionGAN = GANLoss()  # 判别gan
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss_torch()
            self.n_layers_D = 3
            self.num_D = 2
        elif self.pattern == 'MAP':  # 创建mapping模型
            from .mapping_modules.Mapping import Map
            from .discriminator.networks import MultiscaleDiscriminator
            # from models.VAE.autoencoder.autoencoder import AautoencoderKL as UNet
            from .discriminator.networks import GANLoss, VGGLoss_torch
            self.denoise_fn = Map(module_name, model_parameter['unet_A'],
                                  model_parameter['mapping'])
            self.discriminator_y_0 = MultiscaleDiscriminator(3, 64, 3,
                                                             functools.partial(nn.InstanceNorm2d, affine=False),
                                                             False, 2,
                                                             True)
            #####################
            self.criterionGAN = GANLoss()  # 判别gan
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss_torch()
            self.n_layers_D = 3
            self.num_D = 2
        elif self.pattern == 'NP':  # 创建mapping模型
            from .mapping_modules.Mapping import Map
            from models.VAE.autoencoder.autoencoder import AautoencoderKL as UNet
            from models.NPNet.NPNet import NPNet
            self.denoise_fn = Map(module_name, model_parameter['unet_A'], model_parameter['unet_B'],
                                  model_parameter['mapping'])
            self.NPNet = NPNet()
        elif self.pattern == 'D':
            from .detection.unet import UNet
            self.denoise_fn = UNet(**model_parameter['detection'])
        elif self.pattern == 'R':
            from .mapping_modules.Mapping import Map
            from models.VAE.autoencoder.autoencoder import AautoencoderKL as UNet
            self.denoise_fn = Map(module_name, model_parameter['unet_A'],
                                  model_parameter['mapping'])
        else:
            raise Exception(f"Not pattern parameter: {self.pattern}")
        self.beta_schedule = beta_schedule  # 时间调度的beta相关参数
        self.lambda_feat = 10.0

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    # 设置噪声调度方案
    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)  # 固定参数
        betas = make_beta_schedule(**self.beta_schedule[phase])  # 创建betas
        betas = betas.detach().cpu().numpy() if isinstance(  # 将betas放在CPU上
            betas, torch.Tensor) else betas  #
        alphas = 1. - betas  # alphas
        timesteps, = betas.shape  # 获取加燥步数
        self.num_timesteps = int(timesteps)
        # gammas=alphas_cumprod
        gammas = np.cumprod(alphas, axis=0)  # alphas累乘
        # gammas_prev=alphas_cumprod_prev
        gammas_prev = np.append(1., gammas[:-1])  #
        gammas_next = np.append(gammas[1:], 0.0)  #

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('gammas_prev', to_torch(gammas_prev))
        self.register_buffer('gammas_next', to_torch(gammas_next))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    ################重构set_new_noise_schedule#############################
    def reset_new_noise_schedule(self, device=torch.device('cuda'), phase=None):
        if phase == 'test' or phase == 'valid' or self.pattern == 'NP':
            to_torch = partial(torch.tensor, dtype=torch.float32, device=device)  # 固定参数
            use_timesteps = space_timesteps(self.num_timesteps, self.spacing)
            timestep_map = []  # 映射模型需要的时间节点
            last_alpha_cumprod = 1.0  # 上一个alpha_cumprod
            new_betas = []
            for i, alpha_cumprod in enumerate(self.gammas):
                if i in use_timesteps:
                    # 来自beta与alpha之间的关系
                    new_betas.append((1 - alpha_cumprod / last_alpha_cumprod).cpu().detach().numpy())
                    last_alpha_cumprod = alpha_cumprod
                    timestep_map.append(i)

            betas = np.array(new_betas)
            alphas = 1. - betas  # alphas

            timesteps, = betas.shape  # 获取加燥步数
            self.num_timesteps = int(timesteps)
            # gammas=alphas_cumprod
            gammas = np.cumprod(alphas, axis=0)  # alphas累乘

            # gammas_prev=alphas_cumprod_prev
            gammas_prev = np.append(1., gammas[:-1])  #
            gammas_next = np.append(gammas[1:], 0.0)  #

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.register_buffer('gammas', to_torch(gammas))
            self.register_buffer('gammas_prev', to_torch(gammas_prev))
            self.register_buffer('gammas_next', to_torch(gammas_next))
            self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
            self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
            # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
            self.register_buffer('posterior_log_variance_clipped',
                                 to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
            self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
            self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    ######################################################################
    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def predict_noise_from_start(self, y_t, t, start):
        # eps*sqrt(1-alpha_cum[t])+sqrt(alpha_cum[t])*x[0]=x[t]
        # 推导：eps=x[t]/sqrt(1-alpha_cum[t])-x[0]/sqrt(1/alpha_cum[t]-1)
        # 化简后可以得到
        # eps=(1/sqrt(alpha_cum[t])*x[t]-x[0])/sqrt(1/alpha_cum[t]-1)
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
                - start
        ) / extract(self.sqrt_recipm1_gammas, t, y_t.shape)

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, mask=None, gammas=None, inst=None):  # y_t就是y[t]
        if gammas is not None:
            noise_level = gammas
        else:
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        if y_cond is not None:
            y_0_hat = self.denoise_fn(x=torch.cat((y_cond, y_t), dim=1), gammas=noise_level, mask=mask, inst=inst)
        else:
            y_0_hat = self.denoise_fn(x=y_t, gammas=noise_level, mask=mask, inst=inst)
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance, y_0_hat

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt() * y_0 +
                (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()  #
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, mask=None, gammas=None):
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(  # 获得均值和方差
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, mask=mask, gammas=gammas)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp(), y_0_hat  # 采样x[t-1]

    @torch.no_grad()
    def p_ddim_sample(self, y_t, t, clip_denoised=True, y_cond=None, eta=0.0, mask=None, gammas=None, inst=None):
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(  # 获得均值和方差
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, mask=mask, gammas=gammas, inst=inst)
        eps = self.predict_noise_from_start(y_t, t, y_0_hat)  # x[t]、预测的x[0]得到噪声
        alpha_bar = extract(self.gammas, t, y_t.shape)
        alpha_bar_prev = extract(self.gammas_prev, t, y_t.shape)
        sigma = (
                eta  # 标准的DDIM
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (  # 计算均值
                y_0_hat * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return mean_pred + noise * sigma, y_0_hat  # 采样x[t-1]

    @torch.no_grad()
    def ddim_reverse_sample(self, y_t, t, clip_denoised=True, y_cond=None, eta=0.0, mask=None, gammas=None):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(  # 获得均值和方差
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, mask=mask, gammas=gammas)

        eps = self.predict_noise_from_start(y_t, t, y_0_hat)  # x[t]、预测的x[0]得到噪声

        alpha_bar_next = extract(self.gammas_next, t, y_t.shape)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        mean_pred = (
                y_0_hat * torch.sqrt(alpha_bar_next)
                + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def testA(self, g_t, synthetic, mask, mask_abs, inst):
        b, *_ = g_t.shape  # 获得batchsize
        mask_mean_values = torch.mean(mask, dim=(1, 2, 3), keepdim=True)  # 保持维度
        mask_mean = torch.where(mask_abs == 1, mask_abs, mask_mean_values)
        cond_image = synthetic * (1 - mask_abs) + mask_abs * torch.randn_like(synthetic)
        t = torch.randint(0, self.num_timesteps, (b,), device=g_t.device).long()
        flag = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) > (mask_mean_values * self.num_timesteps).int()
        sample_gammas = self.get_gammas(b, t, g_t.device)
        noise = torch.randn_like(g_t)  # 没有噪声，创建一个噪声
        y_noise = self.q_sample(  # xt
            y_0=synthetic, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        # y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=mask)
        y_A_input = synthetic * (1. - mask) + mask * y_noise
        # y_A_input = torch.where(flag, synthetic * (1. - mask_abs) + mask_abs * y_noise.clamp_(-1.0, 1.0),
        #                         synthetic * (1. - mask_mean) + mask_mean * y_noise.clamp_(-1.0, 1.0))

        output = self.denoise_fn(torch.cat((cond_image, y_A_input), dim=1), imgc=torch.zeros_like(inst))
        return output, torch.cat((synthetic, cond_image, y_A_input, output), dim=0)

    @torch.no_grad()
    def testB(self, y_0, synthetic, mask, mask_synthetic, sample_num, inst):
        b, *_ = y_0.shape  # 获得batchsize
        y_0_hat = self.denoise_fn(x=y_0, imgc=inst)
        return y_0_hat, torch.cat([y_0, y_0_hat])

    @torch.no_grad()
    def ddim_reverse_sample_progressive(self, y_0, synthetic, mask, mask_abs, mask_synthetic):
        b, *_ = synthetic.shape  # 条件图像的大小
        # sample_inter = (self.num_timesteps // sample_num)  # 记录去噪图片的步长
        # assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        ret_arr = y_0
        y_t = y_0
        # x[0]->x[T]
        # from utils.showImage import Show
        for i in tqdm(range(0, self.num_timesteps), desc='ddim reverse sample loop time step', total=self.num_timesteps,
                      leave=False, position=1, mininterval=10):
            t = torch.full((b,), i, device=y_t.device, dtype=torch.long)  # 获取t
            gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=gammas.view(-1, 1, 1, 1), noise=mask)
            y_t = synthetic * (1. - mask) + mask * y_t.clamp_(-1.0, 1.0)
            # y_A_input = torch.where(mask_abs.bool(), synthetic * (1. - mask) + mask * y_noise,synthetic * (1. - mask) + mask * y_synthetic)
            y_t = self.ddim_reverse_sample(y_t, t, y_cond=torch.cat((y_mask, mask_synthetic), dim=1), mask=mask_abs,
                                           gammas=gammas)
            # if i % sample_inter == 0:  # 存档图片
            #     ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    @torch.no_grad()
    def testMAP(self, y_0, synthetic, mask, mask_abs, mask_synthetic, sample_num, inst):
        b, *_ = synthetic.shape  # 条件图像的大小
        # mask_mean_values = torch.mean(mask, dim=(1, 2, 3), keepdim=True)  # 保持维度
        # mask_mean = torch.where(mask_abs == 1, mask_abs, mask_mean_values)
        # cond_image = synthetic * (1 - mask_abs) + mask_abs * torch.randn_like(synthetic)
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)  # 记录去噪图片的步长
        y_t = torch.randn_like(synthetic)
        # y_A_input = synthetic * (1. - mask) + mask * y_t.clamp_(-1.0, 1.0)
        # noise = torch.randn_like(y_0)

        # sample_gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # ret_arr = cond_image
        # x[T]->x[0]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps,
                      leave=False, position=1, mininterval=10):
            t = torch.full((b,), i, device=mask.device, dtype=torch.long)  # 获取t
            # flag = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) > (mask_mean_values * self.num_timesteps).int()
            # y_t = torch.where(flag, synthetic * (1. - mask_abs) + mask_abs * y_t.clamp_(-1.0, 1.0),
            #                   synthetic * (1. - mask_mean) + mask_mean * y_t.clamp_(-1.0, 1.0))
            y_t = synthetic * (1. - mask_abs) + mask_abs * y_t.clamp_(-1.0, 1.0)

            gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            # y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=gammas.view(-1, 1, 1, 1), noise=mask)
            if self.sample_function == "ddpm":
                y_t, y_0_hat = self.p_sample(y_t, t, y_cond=mask, mask=mask_abs, gammas=gammas,
                                             inst=torch.zeros_like(inst))  # 一次p_sample采样，传入x_t和x_cond
            else:
                y_t, y_0_hat = self.p_ddim_sample(y_t, t, y_cond=mask, mask=mask_abs, gammas=gammas,
                                                  inst=torch.zeros_like(inst))
            if i % sample_inter == 0:  # 存档图片
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            # assert not torch.all(mask == 0), "there is not mask"
        ret_arr = torch.cat([ret_arr, (mask * 2 - 1).repeat(1, 3, 1, 1), synthetic, y_0], dim=0)

        return y_t, ret_arr

    @torch.no_grad()
    def testNP(self, y_0, synthetic, mask, mask_abs, mask_synthetic, sample_num, y_t=None):
        b, *_ = synthetic.shape  # 条件图像的大小
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)  # 记录去噪图片的步长
        Init_X_T = torch.randn_like(mask_synthetic)
        # create NPNet to get the target noise
        golden_noise = self.NPNet.reference(Init_X_T=Init_X_T, mask=mask, mask_synthetic=mask_synthetic)

        y_t = golden_noise
        ret_arr = golden_noise
        # x[T]->x[0]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps,
                      leave=False, position=1, mininterval=10):
            t = torch.full((b,), i, device=mask.device, dtype=torch.long)  # 获取t
            y_t = synthetic * (1. - mask) + mask * y_t.clamp_(-1.0, 1.0)

            gammas = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=gammas.view(-1, 1, 1, 1), noise=mask)
            if self.sample_function == "ddpm":
                y_t, y_0_hat = self.p_sample(y_t, t, y_cond=torch.cat((y_mask, mask_synthetic), dim=1), mask=mask_abs,
                                             gammas=gammas)  # 一次p_sample采样，传入x_t和x_cond
            else:
                y_t, y_0_hat = self.p_ddim_sample(y_t, t, y_cond=y_mask, mask=mask,
                                                  gammas=gammas)
            if i % sample_inter == 0:  # 存档图片
                ret_arr = torch.cat([ret_arr, y_0_hat], dim=0)
        ret_arr = torch.cat([ret_arr, (mask * 2 - 1).repeat(1, 3, 1, 1), synthetic, y_0], dim=0)
        return y_t, ret_arr

    @torch.no_grad()
    def testD(self, y_0, synthetic, mask, sample_num):
        mask_hat = self.denoise_fn(synthetic).repeat(1, 3, 1, 1)
        return torch.maximum(torch.clamp(mask_hat, 0, 200 / 255) * 2 - 1, mask), torch.cat(
            (synthetic, mask_hat * 2 - 1, (mask * 2 - 1).repeat(1, 3, 1, 1)), dim=0)

    @torch.no_grad()
    def restoration(self, y_0, synthetic, mask, mask_abs, mask_synthetic, sample_num, inst):
        if self.pattern == 'MAP' or self.pattern == 'AandMAP':
            return self.testMAP(y_0, synthetic, mask, mask_abs, mask_synthetic, sample_num=sample_num,
                                inst=inst)
        elif self.pattern == 'NP':
            return self.testNP(y_0, synthetic, mask, mask_abs, mask_synthetic, sample_num=sample_num)
        elif self.pattern == 'A':
            return self.testA(y_0, synthetic, mask, mask_abs=mask_abs, inst=inst)
        elif self.pattern == 'B':
            return self.testB(y_0, synthetic, mask, mask_synthetic, sample_num=sample_num, inst=inst)
        elif self.pattern == 'D':
            return self.testD(y_0, synthetic, mask, sample_num=sample_num)
        elif self.pattern == 'R':
            y_t, ret_arr = self.ddim_reverse_sample_progressive(y_0, synthetic, mask, mask_synthetic,
                                                                sample_num=sample_num)  # 获得目标的x_t

            y_t, ret_arr2 = self.testMAP(y_0=y_0, synthetic=synthetic, mask=mask, mask_abs=mask_abs,
                                         mask_synthetic=mask_synthetic,
                                         sample_num=sample_num, y_t=y_t)

            y_t, ret_arr3 = self.testMAP(y_0=y_0, synthetic=synthetic, mask=mask, mask_abs=mask_abs,
                                         mask_synthetic=mask_synthetic,
                                         sample_num=sample_num, y_t=None)
            return y_t, torch.cat((ret_arr, (y_t - synthetic * (1. - mask)) / mask, ret_arr2, ret_arr3), dim=0)

    def get_gammas(self, b, t, device):
        gamma_t1 = extract(self.gammas_prev, t, x_shape=(1, 1))  # 索引t-1对应的gamma
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=device) + gamma_t1
        return sample_gammas.view(b, -1)

    # 训练的一次前向传播过程
    def forward(self, y_0, synthetic, mask, mask_synthetic, inst, infer=False):
        # sampling from p(gammas)
        if self.pattern == 'MAP':
            loss, generator = self.trainMAP(y_0=y_0, synthetic=synthetic, mask=mask, inst=inst, infer=infer)
            return loss, generator if infer else None
        elif self.pattern == 'B':
            loss, generator = self.trainB(y_0=y_0, synthetic=synthetic, mask=mask, inst=inst, infer=infer)
            return loss, generator if infer else None
        elif self.pattern == 'D':
            mask_hat = self.denoise_fn(synthetic).repeat(1, 3, 1, 1)
            loss = self.loss_fn(loss_name='mse', output=mask_hat, target=mask, mask=mask)
            return loss, torch.cat((synthetic, mask_hat * 2 - 1, (mask * 2 - 1).repeat(1, 3, 1, 1))) if infer else None

    def trainA(self, g_t, input, mask, mask_synthetic, inst, infer, optD_syn, optD_y_0, optD_feat, optD_Contra_gan):
        # input等与syn
        b, *_ = input.shape  # 获得batchsize
        mask_abs = torch.where(mask == 1, torch.tensor(1), torch.tensor(0)).float()
        mask_mean_values = torch.mean(mask, dim=(1, 2, 3), keepdim=True)  # 保持维度
        mask_mean = torch.where(mask_abs == 1, mask_abs, mask_mean_values)
        cond_image = input * (1 - mask_abs) + mask_abs * torch.randn_like(input)
        # print(mask_mean_values)
        loss_dict = {}

        random_t = torch.randint(0, self.num_timesteps, (1,), device=input.device).long()
        t = random_t.expand(b)
        # 决定分界线
        flag = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) > (mask_mean_values * self.num_timesteps).int()

        sample_gammas = self.get_gammas(b, t, input.device)
        noise = torch.randn_like(input)  # 没有噪声，创建一个噪声
        # y_noisy = self.q_sample(  # xt
        #     y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        y_synthetic = self.q_sample(  # xt
            y_0=input, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        # y_A_input = input * (1. - mask) + mask * y_synthetic
        #
        # y_A_input = torch.where(flag,
        #                         input * (1. - mask_abs) + mask_abs * y_synthetic,
        #                         input * (1. - mask_mean) + mask_mean * y_synthetic)

        y_A_input = input * (1. - mask_abs) + mask_abs * y_synthetic
        target = input
        # y_A_input = input * (1. - mask) + mask * y_synthetic
        hidden = self.denoise_fn.encodeing(y_A_input)
        #####################################################
        # #
        # sup_and_contra_inst = inst.clone()  # 创建 inst 的副本，以保持原始数据不变
        # proxy = self.denoise_fn.get_class_embedding(inst)
        # # sup_and_contra_inst[sup_and_contra_inst == 1] = 0
        # # # print('sup_and_contra_inst', sup_and_contra_inst)
        # # # # # Super loss
        # features = self.SupCon(random_latent)
        # #
        # sup_loss = self.SupConLoss(embed=features, proxy=proxy,
        #                            label=sup_and_contra_inst)
        # loss_dict.update({'Sup_loss': loss_dict.get('Sup_loss', 0) + sup_loss.mean()})
        #######################################################
        # noise = Variable(torch.randn(hidden.size()).cuda(hidden.data.get_device()))
        out_put = self.denoise_fn.decodeing(hidden)  # 输出syn和y_A_input
        mask_out_put = torch.where(mask_abs.bool(), torch.tensor(-1.0), out_put)
        # real_old_feat = random_latent[inst == 0]  # 真图像潜码特征
        # syn_feat = random_latent[inst == 1]  # 合成图像潜码特征

        fake_synthetic = mask_out_put[inst != 2]  # mask_out_put[inst != 2]  # 输入合成和旧图像,(去除了mask部分)
        # fake_y_0 = out_put[inst != 2]  # mask_out_put[inst == 2]  # 重建干净图像，(去除了mask部分)
        synthetic = mask_synthetic[inst != 2]  # 输入的合成和真实旧图片，(去除了mask部分)
        # y_0 = target[inst == 2]  # 输入干净图像，(去除了mask部分)
        #
        syn_weight = 1  # synthetic.shape[0] / b
        # y_0_weight = y_0.shape[0] / b
        #######################################################
        # L = min(len(real_old_feat), len(syn_feat))  # 统计两边哪边最小
        # real_old_feat = real_old_feat[:L]  # 取出同样最小的部分
        # syn_feat = syn_feat[:L]
        #
        # pred_fake_feat = self.feat_discriminate(real_old_feat)  # 真实图像的潜码输入特征判别器，得到的各层特征值
        # loss_featD_fake = self.criterionGAN(pred_fake_feat, False)  # 计算判别器判别合成数据为假的损失
        # loss_dict.update({'featD_fake': loss_dict.get('featD_fake', 0) + loss_featD_fake.mean()})
        #
        # pred_real_feat = self.feat_discriminate(syn_feat)  # 合成图像输入特征判别器
        # loss_featD_real = self.criterionGAN(pred_real_feat, True)  # 计算判别其判别真实老照片数据为真的损失
        # loss_dict.update({'featD_real': loss_dict.get('featD_real', 0) + loss_featD_real.mean()})
        # #####特征判别器先更新
        # loss_D_feat = loss_featD_fake + loss_featD_real
        # optD_feat.zero_grad()
        # loss_D_feat.backward()
        # optD_feat.step()
        # # #############Generator对生成特征的损失####################
        # pred_fake_feat_G = self.discriminator_feat.forward(real_old_feat)  # 这个操作和pred_fake_feat一样
        # loss_G_featD = self.criterionGAN(pred_fake_feat_G, True)  # 判别器判断正确的损失
        # loss_dict.update({'G_featD': loss_dict.get('G_featD', 0) + loss_G_featD.mean()})
        # # # ##############syn图像层面的损失#######################################################
        pred_fake_syn_pool = self.syn_discriminate(fake_synthetic)
        loss_D_syn_fake = self.criterionGAN(pred_fake_syn_pool, False)
        loss_dict.update({'D_fake': loss_dict.get('D_fake', 0) + loss_D_syn_fake.mean() * syn_weight})
        # Real Detection and Loss
        pred_real_syn = self.syn_discriminate(synthetic)
        loss_D_syn_real = self.criterionGAN(pred_real_syn, True)
        loss_dict.update({'D_real': loss_dict.get('D_real', 0) + loss_D_syn_real.mean() * syn_weight})
        #######图像判别器先更新
        loss_D_syn = loss_D_syn_fake + loss_D_syn_real
        optD_syn.zero_grad()
        loss_D_syn.backward()
        optD_syn.step()
        #################Generator的损失########################
        # # # GAN loss (Fake Passability Loss)
        pred_fake_syn = self.discriminator_syn.forward(fake_synthetic)
        loss_G_syn_GAN = self.criterionGAN(pred_fake_syn, True)
        loss_dict.update({'G_GAN': loss_dict.get('G_GAN', 0) + loss_G_syn_GAN.mean() * syn_weight})
        # GAN feature matching loss
        loss_G_syn_GAN_Feat = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(pred_fake_syn[i]) - 1):
                loss_G_syn_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake_syn[i][j],
                                                          pred_real_syn[i][j].detach()) * self.lambda_feat
        loss_dict.update({'G_GAN_Feat': loss_dict.get('G_GAN_Feat', 0) + loss_G_syn_GAN_Feat.mean() * syn_weight})
        # # ########################################################################################
        # # # ##############y_0图像层面的损失##############
        # # pred_fake_y_0_pool = self.y_0_discriminate(fake_y_0)
        # # loss_D_y_0_fake = self.criterionGAN(pred_fake_y_0_pool, False)
        # # loss_dict.update({'D_fake': loss_dict.get('D_fake', 0) + loss_D_y_0_fake.mean() * y_0_weight})
        # #
        # # # Real Detection and Loss
        # # pred_real_y_0 = self.y_0_discriminate(y_0)
        # # loss_D_y_0_real = self.criterionGAN(pred_real_y_0, True)
        # # loss_dict.update({'D_real': loss_dict.get('D_real', 0) + loss_D_y_0_real.mean() * y_0_weight})
        # #
        # # loss_D_y_0 = loss_D_y_0_fake + loss_D_y_0_real
        # # optD_y_0.zero_grad()
        # # loss_D_y_0.backward()
        # # optD_y_0.step()
        # #
        # # # GAN loss (Fake Passability Loss)
        # # pred_fake_y_0 = self.discriminator_y_0.forward(fake_y_0)
        # # loss_G_GAN = self.criterionGAN(pred_fake_y_0, True)
        # # loss_dict.update({'G_GAN': loss_dict.get('G_GAN', 0) + loss_G_GAN.mean() * y_0_weight})
        # # # # GAN feature matching loss
        # # # loss_G_y_0_GAN_Feat = 0
        # # # feat_weights = 4.0 / (self.n_layers_D + 1)
        # # # D_weights = 1.0 / self.num_D
        # # # for i in range(self.num_D):
        # # #     for j in range(len(pred_fake_y_0[i]) - 1):
        # # #         loss_G_y_0_GAN_Feat += D_weights * feat_weights * \
        # # #                                self.criterionFeat(pred_fake_y_0[i][j],
        # # #                                                   pred_real_y_0[i][j].detach()) * self.lambda_feat
        # # # loss_dict.update({'G_GAN_Feat': loss_dict.get('G_GAN_Feat', 0) + loss_G_y_0_GAN_Feat.mean() * y_0_weight})
        # # ############################################################################
        # # # # # ContraltiveGan
        # # # real_contr_loss = self.discriminator_contra_gan(x=mask_synthetic, label=sup_and_contra_inst)
        # # # fake_contr_loss_pool = self.discriminator_contra_gan(x=mask_out_put.detach(), label=sup_and_contra_inst)
        # # # loss_D_l_2c_loss = self.criterionConditionalContrastiveLoss(embed=real_contr_loss['embed'],
        # # #                                                             proxy=proxy.detach(),
        # # #                                                             label=sup_and_contra_inst)  # 判别器输入
        # # # loss_D_contral_adv = torch.mean(F.relu(1. - real_contr_loss['adv_output'])) + torch.mean(
        # # #     F.relu(1. + fake_contr_loss_pool['adv_output']))
        # # # loss_D_contral = loss_D_l_2c_loss + loss_D_contral_adv
        # # # optD_Contra_gan.zero_grad()
        # # # loss_D_contral.backward()
        # # # optD_Contra_gan.step()
        # # # loss_dict.update({'ConraD_loss': loss_dict.get('ConraD_loss', 0) + loss_D_contral.mean()})
        # # #
        # # # # GAN损失
        # # # fake_contr_loss = self.discriminator_contra_gan(x=mask_out_put, label=sup_and_contra_inst)
        # # # loss_G_contral_l_2c = self.criterionConditionalContrastiveLoss(embed=fake_contr_loss['embed'],
        # # #                                                                proxy=proxy,
        # # #                                                                label=sup_and_contra_inst)  # 判别器输入
        # # # loss_G_contral_gan = -torch.mean(fake_contr_loss['adv_output'])
        # # # loss_G_contral = loss_G_contral_l_2c + loss_G_contral_gan
        # # # loss_dict.update({'G_GAN_CG': loss_dict.get('G_GAN_CG', 0) + loss_G_contral.mean()})
        # # ###########################################################################
        # # # VGG feature matching loss
        # # loss_G_syn_VGG = self.criterionVGG(out_put, target) * self.lambda_feat
        # # loss_dict.update({'G_VGG': loss_dict.get('G_VGG', 0) + loss_G_syn_VGG.mean()})
        # ###########################################################################
        # loss_G_kl = torch.mean(torch.pow(random_latent, 2))  # KL散度损失
        # loss_dict.update({'G_KL': loss_dict.get('G_KL', 0) + loss_G_kl.mean()})

        smooth_loss = self.loss_fn(output=out_put, target=target) * self.lambda_feat  # 只对真实旧照片求L1 loss

        loss_dict.update({'smooth_loss': loss_dict.get('smooth_loss', 0) + smooth_loss.mean()})
        if infer:
            with torch.no_grad():
                fate_recount = self.denoise_fn(y_A_input)
        return loss_dict, torch.cat((
            mask_synthetic,
            y_A_input,
            fate_recount,
            out_put,
            mask_out_put,
            target
        ), dim=0) if infer else None

    def trainMAP(self, y_0, synthetic, mask, mask_abs, mask_synthetic, inst, infer=False, optD_y_0=None):
        loss_dict = {}
        b, *_ = y_0.shape  # 获得batchsize

        mask_mean_values = torch.mean(mask, dim=(1, 2, 3), keepdim=True)  # 保持维度
        mask_mean = torch.where(mask_abs == 1, mask_abs, mask_mean_values)
        cond_image = synthetic * (1 - mask_abs) + mask_abs * torch.randn_like(synthetic)
        # cond_image = synthetic * (1 - mask_abs) + mask_abs * torch.randn_like(synthetic)
        t = torch.randint(0, self.num_timesteps, (b,), device=y_0.device).long()
        # 决定分界线
        # flag = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) > (mask_mean_values * self.num_timesteps).int()

        sample_gammas = self.get_gammas(b, t, y_0.device)  # 得到有一定扰动的gammas
        noise = torch.randn_like(y_0)  # 没有噪声，创建一个噪声
        # y_synthetic = self.q_sample(  # s_t
        #     y_0=synthetic, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        y_noise = self.q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)  # y_t

        # y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=mask)
        with torch.no_grad():
            # y_A_input = synthetic * (1. - mask) + mask * y_noise
            # y_A_input = torch.where(flag,
            #                         synthetic * (1. - mask_abs) + mask_abs * y_noise,
            #                         synthetic * (1. - mask_mean) + mask_mean * y_noise)
            y_A_input = synthetic * (1. - mask_abs) + mask_abs * y_noise
            y_t_A_en = self.denoise_fn.A.encodeing(y_A_input)

        fake_y_0 = self.denoise_fn.Map(x=y_t_A_en, gammas=sample_gammas, mask=mask)
        mask_y_0 = torch.where(mask_abs.bool(), torch.tensor(-1.0), y_0)
        mask_fake_y_0 = torch.where(mask_abs.bool(), torch.tensor(-1.0), fake_y_0)

        # ##############y_0图像层面的损失##############
        pred_fake_y_0_pool = self.y_0_discriminate(mask_fake_y_0)
        loss_D_y_0_fake = self.criterionGAN(pred_fake_y_0_pool, False)
        loss_dict.update({'D_fake': loss_dict.get('D_fake', 0) + loss_D_y_0_fake.mean()})

        # Real Detection and Loss
        pred_real_y_0 = self.y_0_discriminate(mask_y_0)
        loss_D_y_0_real = self.criterionGAN(pred_real_y_0, True)
        loss_dict.update({'D_real': loss_dict.get('D_real', 0) + loss_D_y_0_real.mean()})

        loss_D_y_0 = loss_D_y_0_fake + loss_D_y_0_real
        optD_y_0.zero_grad()
        loss_D_y_0.backward()
        optD_y_0.step()

        # GAN loss (Fake Passability Loss)
        pred_fake_y_0 = self.discriminator_y_0.forward(mask_fake_y_0)
        loss_G_GAN = self.criterionGAN(pred_fake_y_0, True)
        loss_dict.update({'G_GAN': loss_dict.get('G_GAN', 0) + loss_G_GAN.mean()})
        # GAN feature matching loss
        loss_G_y_0_GAN_Feat = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(pred_fake_y_0[i]) - 1):
                loss_G_y_0_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake_y_0[i][j],
                                                          pred_real_y_0[i][j].detach()) * self.lambda_feat * 0.5
        loss_dict.update({'G_GAN_Feat': loss_dict.get('G_GAN_Feat', 0) + loss_G_y_0_GAN_Feat.mean()})
        #
        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(mask_fake_y_0, mask_y_0) * self.lambda_feat
        loss_dict.update({'G_VGG': loss_G_VGG.mean()})

        non_mask_l1loss = self.loss_fn(output=fake_y_0, target=y_0) * self.lambda_feat
        loss_dict.update({'smooth_loss': non_mask_l1loss.mean()})
        ################################
        if infer:
            with torch.no_grad():
                generator = torch.cat(
                    (
                        y_A_input,
                        self.denoise_fn.A.decodeing(y_t_A_en.detach()),  # 输入重建
                        fake_y_0,  # 输出图像
                        y_0),  # 目标
                    dim=0)
        ################################

        return loss_dict, generator if infer else None

    def trainAandMAP(self, y_0, synthetic, mask, mask_abs, mask_synthetic, inst, infer=False,
                     optD_syn=None, optD_y_0=None, optD_feat=None):
        loss_dict = {}
        b, *_ = y_0.shape  # 获得batchsize

        # mask_mean_values = torch.mean(mask, dim=(1, 2, 3), keepdim=True)  # 保持维度
        # mask_mean = torch.where(mask_abs == 1, mask_abs, mask_mean_values)
        cond_image = synthetic * (1 - mask_abs) + mask_abs * torch.randn_like(synthetic)
        t = torch.randint(0, self.num_timesteps, (b,), device=y_0.device).long()
        # 决定分界线
        # flag = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) > (mask_mean_values * self.num_timesteps).int()

        sample_gammas = self.get_gammas(b, t, y_0.device)  # 得到有一定扰动的gammas
        noise = torch.randn_like(y_0)  # 没有噪声，创建一个噪声
        # y_synthetic = self.q_sample(  # s_t
        #     y_0=synthetic, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        y_noise = self.q_sample(y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)  # y_t

        # y_mask = self.q_sample(y_0=torch.zeros_like(mask), sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=mask)
        # with torch.no_grad():
        # # y_A_input = synthetic * (1. - mask) + mask * y_noise
        # y_A_input = torch.where(flag,
        #                             synthetic * (1. - mask_abs) + mask_abs * y_noise,
        #                             synthetic * (1. - mask_mean) + mask_mean * y_noise)
        y_A_input = synthetic * (1. - mask_abs) + mask_abs * y_noise
        target_A = synthetic
        y_t_A_en = self.denoise_fn.A.encodeing(x=torch.cat((cond_image, y_A_input), dim=1), gammas=sample_gammas,
                                               imgc=inst)
        #####################################################
        # #
        # sup_and_contra_inst = inst.clone()  # 创建 inst 的副本，以保持原始数据不变
        # proxy = self.denoise_fn.A.get_embedding(inst)
        # # sup_and_contra_inst[sup_and_contra_inst == 1] = 0
        # # # print('sup_and_contra_inst', sup_and_contra_inst)
        # # # # # Super loss
        # features = self.SupCon(y_t_A_en)
        # #
        # sup_loss = self.SupConLoss(embed=features, proxy=proxy,
        #                            label=sup_and_contra_inst)
        # loss_dict.update({'Sup_loss': loss_dict.get('Sup_loss', 0) + sup_loss.mean()})
        #######################################################
        # noise = Variable(torch.randn(hidden.size()).cuda(hidden.data.get_device()))
        fake_A = self.denoise_fn.A.decodeing(h=y_t_A_en, imgc=sample_gammas)
        mask_fake_A = torch.where(mask_abs == 1, torch.tensor(-1), fake_A)
        with torch.no_grad():
            y_MAP_input = self.denoise_fn.A.encodeing(x=torch.cat((cond_image[inst == 1], y_A_input[inst == 1]), dim=1),
                                                      imgc=sample_gammas[inst == 1])
        # 修复的输出
        fake_y_0 = self.denoise_fn.Map(x=y_MAP_input, gammas=sample_gammas[inst == 1], mask=mask_abs[inst == 1])
        real_old_feat = y_t_A_en[inst == 0]  # 真图像潜码特征
        syn_feat = y_t_A_en[inst == 1]  # 合成图像潜码特征
        #######################################################
        L = min(len(real_old_feat), len(syn_feat))  # 统计两边哪边最小
        real_old_feat = real_old_feat[:L]  # 取出同样最小的部分
        syn_feat = syn_feat[:L]

        pred_fake_feat = self.feat_discriminate(real_old_feat)  # 真实图像的潜码输入特征判别器，得到的各层特征值
        loss_featD_fake = self.criterionGAN(pred_fake_feat, False)  # 计算判别器判别合成数据为假的损失
        loss_dict.update({'featD_fake': loss_dict.get('featD_fake', 0) + loss_featD_fake.mean()})

        pred_real_feat = self.feat_discriminate(syn_feat)  # 合成图像输入特征判别器
        loss_featD_real = self.criterionGAN(pred_real_feat, True)  # 计算判别其判别真实老照片数据为真的损失
        loss_dict.update({'featD_real': loss_dict.get('featD_real', 0) + loss_featD_real.mean()})
        #####特征判别器先更新
        loss_D_feat = loss_featD_fake + loss_featD_real
        optD_feat.zero_grad()
        loss_D_feat.backward()
        optD_feat.step()
        # #############Generator对生成特征的损失####################
        pred_fake_feat_G = self.discriminator_feat.forward(real_old_feat)  # 这个操作和pred_fake_feat一样
        loss_G_featD = self.criterionGAN(pred_fake_feat_G, True)  # 判别器判断正确的损失
        loss_dict.update({'G_featD': loss_dict.get('G_featD', 0) + loss_G_featD.mean()})

        # # ##############syn图像层面的损失#######################################################
        # pred_fake_syn_pool = self.syn_discriminate(mask_fake_A[inst != 2])
        # loss_D_syn_fake = self.criterionGAN(pred_fake_syn_pool, False)
        # loss_dict.update({'D_fake': loss_dict.get('D_fake', 0) + loss_D_syn_fake.mean()})
        # # Real Detection and Loss
        # pred_real_syn = self.syn_discriminate(mask_synthetic[inst != 2])
        # loss_D_syn_real = self.criterionGAN(pred_real_syn, True)
        # loss_dict.update({'D_real': loss_dict.get('D_real', 0) + loss_D_syn_real.mean()})
        # #######图像判别器先更新
        # loss_D_syn = loss_D_syn_fake + loss_D_syn_real
        # optD_syn.zero_grad()
        # loss_D_syn.backward()
        # optD_syn.step()
        # # # #################Generator的损失########################
        # # # # GAN loss (Fake Passability Loss)
        # pred_fake_syn = self.discriminator_syn.forward(mask_fake_A[inst != 2])
        # loss_G_syn_GAN = self.criterionGAN(pred_fake_syn, True)
        # loss_dict.update({'G_A_GAN': loss_dict.get('G_A_GAN', 0) + loss_G_syn_GAN.mean()})
        # # GAN feature matching loss
        # loss_G_syn_GAN_Feat = 0
        # feat_weights = 4.0 / (self.n_layers_D + 1)
        # D_weights = 1.0 / self.num_D
        # for i in range(self.num_D):
        #     for j in range(len(pred_fake_syn[i]) - 1):
        #         loss_G_syn_GAN_Feat += D_weights * feat_weights * \
        #                                self.criterionFeat(pred_fake_syn[i][j],
        #                                                   pred_real_syn[i][j].detach()) * self.lambda_feat
        # loss_dict.update({'G_A_GAN_Feat': loss_dict.get('G_A_GAN_Feat', 0) + loss_G_syn_GAN_Feat.mean()})

        # ##############y_0图像层面的损失##############
        # pred_fake_y_0_pool = self.y_0_discriminate(fake_y_0)
        # loss_D_y_0_fake = self.criterionGAN(pred_fake_y_0_pool, False)
        # loss_dict.update({'D_fake': loss_dict.get('D_fake', 0) + loss_D_y_0_fake.mean()})
        #
        # # Real Detection and Loss
        # pred_real_y_0 = self.y_0_discriminate(y_0[inst == 1])
        # loss_D_y_0_real = self.criterionGAN(pred_real_y_0, True)
        # loss_dict.update({'D_real': loss_dict.get('D_real', 0) + loss_D_y_0_real.mean()})
        #
        # loss_D_y_0 = loss_D_y_0_fake + loss_D_y_0_real
        # optD_y_0.zero_grad()
        # loss_D_y_0.backward()
        # optD_y_0.step()
        #
        # # GAN loss (Fake Passability Loss)
        # pred_fake_y_0 = self.discriminator_y_0.forward(fake_y_0)
        # loss_G_GAN = self.criterionGAN(pred_fake_y_0, True)
        # loss_dict.update({'G_y_0_GAN': loss_dict.get('G_y_0_GAN', 0) + loss_G_GAN.mean()})
        # # GAN feature matching loss
        # loss_G_y_0_GAN_Feat = 0
        # feat_weights = 4.0 / (self.n_layers_D + 1)
        # D_weights = 1.0 / self.num_D
        # for i in range(self.num_D):
        #     for j in range(len(pred_fake_y_0[i]) - 1):
        #         loss_G_y_0_GAN_Feat += D_weights * feat_weights * \
        #                                self.criterionFeat(pred_fake_y_0[i][j],
        #                                                   pred_real_y_0[i][j].detach()) * self.lambda_feat
        # loss_dict.update({'G_y_0_GAN_Feat': loss_dict.get('G_y_0_GAN_Feat', 0) + loss_G_y_0_GAN_Feat.mean()})

        loss_G_kl = torch.mean(torch.pow(y_t_A_en, 2))  # KL散度损失
        loss_dict.update({'G_KL': loss_dict.get('G_KL', 0) + loss_G_kl.mean()})

        # VGG feature matching loss
        # loss_G_A_VGG = self.criterionVGG(fake_A, target_A) * self.lambda_feat
        # loss_dict.update({'G_A_VGG': loss_G_A_VGG.mean()})
        #
        # # VGG feature matching loss
        # loss_G_y_0_VGG = self.criterionVGG(fake_y_0, y_0[inst == 1]) * self.lambda_feat
        # loss_dict.update({'G_y_0_VGG': loss_G_y_0_VGG.mean()})

        non_mask_A_l1loss = self.loss_fn(loss_name='l1', output=fake_A, target=target_A, mask=mask) * self.lambda_feat
        loss_dict.update({'smooth_A_loss': non_mask_A_l1loss.mean()})

        non_mask_y_0_l1loss = self.loss_fn(loss_name='l1', output=fake_y_0, target=y_0[inst == 1],
                                           mask=mask) * self.lambda_feat
        loss_dict.update({'smooth_y_0_loss': non_mask_y_0_l1loss.mean()})
        ################################
        if infer:
            if fake_y_0.shape[0] < b:
                fake_out_put = fake_A.clone()[inst == 1] = fake_y_0.clone()
                # 创建一个 clone 的 fake_A
                fake_out_put = fake_A.clone()
                # 获取 inst 中值为 1 的索引
                indices = (inst == 1)
                # 使用索引替换
                fake_out_put[indices] = fake_y_0
            with torch.no_grad():
                generator = torch.cat(
                    (synthetic,
                     mask_synthetic,
                     y_A_input,
                     self.denoise_fn.A.decodeing(h=y_t_A_en.detach(), imgc=sample_gammas),  # 输入重建
                     fake_out_put,  # 输出图像
                     y_0),  # 目标
                    dim=0)
        ################################

        return loss_dict, generator if infer else None

    def trainNP(self, y_0, synthetic, mask, mask_abs, mask_synthetic, inst, infer=False):
        loss_dict = {}
        with torch.no_grad():
            X_T_prime, ret_arr = self.ddim_reverse_sample_progressive(y_0, synthetic, mask, mask_abs,
                                                                      mask_synthetic)  # 获得目标的x_t
        Init_X_T = torch.randn_like(X_T_prime)
        # create NPNet to get the target noise
        X_T_pred = self.NPNet(Init_X_T, X_T_prime, mask, mask_synthetic)
        mse_l1loss = self.loss_fn(loss_name='l1', output=X_T_pred, target=X_T_prime, mask=mask)
        loss_dict.update({'Mse': mse_l1loss.mean()})
        return loss_dict, X_T_pred if infer else None

    def trainB(self, y_0, synthetic, mask, mask_synthetic, inst, infer=False, optD=None):
        b, *_ = y_0.shape  # 获得batchsize
        posterior = self.denoise_fn.encodeing(x=y_0, imgc=inst)
        hidden = posterior.sample(inference=False)
        fake_y_0 = self.denoise_fn.decodeing(h=hidden, imgc=inst)

        loss_dict = {}
        pred_fake_pool = self.y_0_discriminate(fake_y_0)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        loss_dict.update({'D_fake': loss_D_fake.mean()})

        # Real Detection and Loss
        pred_real = self.y_0_discriminate(y_0)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_dict.update({'D_real': loss_D_real.mean()})

        loss_D = loss_D_fake + loss_D_real
        optD.zero_grad()
        loss_D.backward()
        optD.step()

        # GAN loss (Fake Passability Loss)
        pred_fake = self.discriminator_y_0.forward(fake_y_0)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_dict.update({'G_GAN': loss_G_GAN.mean()})
        #########################################
        # GAN feature matching loss
        # loss_G_GAN_Feat = 0
        # feat_weights = 4.0 / (self.n_layers_D + 1)
        # D_weights = 1.0 / self.num_D
        # for i in range(self.num_D):
        #     for j in range(len(pred_fake[i]) - 1):
        #         loss_G_GAN_Feat += D_weights * feat_weights * \
        #                            self.criterionFeat(pred_fake[i][j],
        #                                               pred_real[i][j].detach()) * self.lambda_feat
        # loss_dict.update({'G_GAN_Feat': loss_G_GAN_Feat.mean()})

        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(fake_y_0, y_0) * self.lambda_feat
        loss_dict.update({'G_VGG': loss_G_VGG.mean()})

        loss = self.loss_fn(loss_name='l1', output=fake_y_0, target=y_0,
                            mask=mask) * self.lambda_feat
        loss_dict.update({'smooth_loss': loss.mean()})

        loss_G_kl = posterior.kl()
        loss_dict.update({'G_KL': loss_G_kl.mean()})

        return loss_dict, torch.cat((y_0, fake_y_0), dim=0) if infer else None

    def feat_discriminate(self, input):  # 判别器
        return self.discriminator_feat.forward(input.detach())

    def syn_discriminate(self, test_image):
        return self.discriminator_syn.forward(test_image.detach())

    def y_t_discriminate(self, test_image):
        return self.discriminator_y_t.forward(test_image.detach())

    def y_0_discriminate(self, test_image):
        return self.discriminator_y_0.forward(test_image.detach())


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)  # 在a张量上最后一维度上用t去索引
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 将张量调整为(b,1,...,1)


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas  # 获得betas列表


if __name__ == '__main__':
    pass
