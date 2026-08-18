import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import os
import torchvision.utils as vutils
from pathlib import Path


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        self.load_networks()  #
        if self.netG.pattern == 'MAP':
            self.optG = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.Map.parameters())),
                **optimizers["Gen"])
            if self.netG.discriminator_y_0 is not None:
                self.optD_y_0 = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_y_0.parameters())),
                    **optimizers["Dis"])
            else:
                self.optD_y_0 = None
        elif self.netG.pattern == 'A':
            self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.parameters())) + \
                                         list(filter(lambda p: p.requires_grad, self.netG.SupCon.parameters())),
                                         **optimizers["Gen"])
            self.optD_syn = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.netG.discriminator_syn.parameters())),
                **optimizers["Dis"])
            if self.netG.discriminator_y_0 is not None:
                self.optD_y_0 = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_y_0.parameters())),
                    **optimizers["Dis"])
            else:
                self.optD_y_0 = None
            self.optD_feat = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.netG.discriminator_feat.parameters())), **optimizers["Dis"])

            if self.netG.discriminator_contra_gan is not None:
                self.optD_cg = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_contra_gan.parameters())),
                    **optimizers["Dis"])
            else:
                self.optD_cg = None
        elif self.netG.pattern == 'AandMAP':  # 准备优化器
            self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.parameters())) + \
                                         list(filter(lambda p: p.requires_grad, self.netG.SupCon.parameters())),
                                         **optimizers["Gen"])  # 优化器加载
            self.optD_syn = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.netG.discriminator_syn.parameters())),
                **optimizers["Dis"])
            if self.netG.discriminator_y_0 is not None:
                self.optD_y_0 = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_y_0.parameters())),
                    **optimizers["Dis"])
            self.optD_feat = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.netG.discriminator_feat.parameters())), **optimizers["Dis"])
            if self.netG.discriminator_contra_gan is not None:
                self.optD_cg = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_contra_gan.parameters())),
                    **optimizers["Dis"])
            else:
                self.optD_cg = None
        elif self.netG.pattern == 'B':
            self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.parameters())),
                                         **optimizers["Gen"])
            if self.netG.discriminator_y_0 is not None:
                self.optD_y_0 = torch.optim.Adam(
                    list(filter(lambda p: p.requires_grad, self.netG.discriminator_y_0.parameters())),
                    **optimizers["Dis"])
            else:
                self.optD_y_0 = None
        elif self.netG.pattern == 'NP':
            self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.NPNet.parameters())),
                                         **optimizers["Gen"])
        else:
            self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.denoise_fn.parameters())),
                                         **optimizers["Gen"])

        self.resume_training()
        self.netG.set_loss(self.loss_fn)
        self.netG.set_new_noise_schedule(phase=self.phase)

        self.netG.reset_new_noise_schedule(phase=kwargs["opt"]["phase"])
        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.gt_image = self.set_device(data.get('gt_image'))
        self.synthetic = self.set_device(data.get('synthetic'))
        self.mask = self.set_device(data.get('mask'))
        self.inst = self.set_device(data.get('inst'))
        self.mask_abs = self.set_device(data.get('mask_abs'))
        self.path = data['path']
        self.batch_size = len(data['path'])
        self.mask_synthetic = self.set_device(torch.where(self.mask_abs.bool(), torch.tensor(-1.0), self.synthetic))


    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT/{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Syn/{}'.format(self.path[idx]))
            ret_result.append(self.synthetic[idx].detach().float().cpu())

            ret_path.append('Out/{}'.format(self.path[idx]))
            ret_result.append(self.output[idx].detach().float().cpu())

            ret_path.append('Mask/{}'.format(self.path[idx]))
            ret_result.append(self.mask[idx].detach().float().cpu() * 2 - 1)

            ret_path.append('Mask_abs/{}'.format(self.path[idx]))
            ret_result.append(torch.where(self.mask[idx].detach().float().cpu() == 1.0, torch.tensor(1.0),
                                          torch.tensor(-1.0)) * 2 - 1)

            ret_path.append('Process/{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Mask_syn/{}'.format(self.path[idx]))
            ret_result.append(self.mask_synthetic[idx::self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path,
                                                       result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        if self.netG.pattern == 'MAP':
            self.netG.denoise_fn.A.eval()
            # self.netG.denoise_fn.B.train()
            self.netG.denoise_fn.Map.train()
            if self.netG.discriminator_y_0 is not None:
                self.netG.discriminator_y_0.train()
        elif self.netG.pattern == 'AandMAP':
            self.netG.denoise_fn.train()
            self.netG.SupCon.train()
            self.netG.discriminator_feat.train()
            self.netG.discriminator_syn.train()
            if self.netG.discriminator_y_0 is not None:
                self.netG.discriminator_y_0.eval()
            if self.netG.discriminator_contra_gan is not None:
                self.netG.discriminator_contra_gan.train()
        elif self.netG.pattern == 'NP':
            self.netG.denoise_fn.A.eval()
            self.netG.denoise_fn.B.eval()
            self.netG.denoise_fn.Map.eval()
            self.netG.NPNet.train()
        elif self.netG.pattern == 'A':
            self.netG.denoise_fn.train()
            self.netG.SupCon.train()
            self.netG.discriminator_feat.train()
            self.netG.discriminator_syn.train()
            if self.netG.discriminator_y_0 is not None:
                self.netG.discriminator_y_0.train()
            if self.netG.discriminator_contra_gan is not None:
                self.netG.discriminator_contra_gan.train()
        elif self.netG.pattern == 'B':
            self.netG.denoise_fn.train()
            if self.netG.discriminator_y_0 is not None:
                self.netG.discriminator_y_0.train()
        else:
            self.netG.denoise_fn.train()
        self.train_metrics.reset()

        for train_data in tqdm.tqdm(self.phase_loader, desc=f'{self.epoch}'):
            infer = self.iter % self.opt['train']['log_iter'] < self.batch_size
            loss_dict = {}
            self.set_input(train_data)
            if self.netG.pattern == 'A':
                loss_dict, generator = self.netG.trainA(self.gt_image, input=self.synthetic, mask=self.mask,
                                                        mask_synthetic=self.mask_synthetic,
                                                        inst=self.inst, infer=infer, optD_syn=self.optD_syn,
                                                        optD_y_0=self.optD_y_0,
                                                        optD_feat=self.optD_feat, optD_Contra_gan=self.optD_cg
                                                        )
                # optD_Contra_gan=self.optD_cg)
                loss_G = loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_GAN', 0) + \
                         loss_dict.get('G_featD', 0) + loss_dict.get('smooth_loss', 0) + loss_dict.get('Sup_loss', 0) + \
                         loss_dict.get('G_GAN_CG', 0) + loss_dict.get('G_KL', 0)

                self.optG.zero_grad()
                loss_G.backward()
                self.optG.step()

                loss = sum(loss_dict.values())
            elif self.netG.pattern == 'AandMAP':
                loss_dict, generator = self.netG.trainAandMAP(self.gt_image, synthetic=self.synthetic, mask=self.mask,
                                                              mask_abs=self.mask_abs,
                                                              mask_synthetic=self.mask_synthetic,
                                                              inst=self.inst, infer=infer, optD_syn=self.optD_syn,
                                                              optD_y_0=self.optD_y_0,
                                                              optD_feat=self.optD_feat  # , optD_Contra_gan=self.optD_cg
                                                              )
                loss_G = loss_dict.get('G_featD', 0) + loss_dict.get('Sup_loss', 0) + \
                         loss_dict.get('G_A_GAN_Feat', 0) + loss_dict.get('G_y_0_GAN_Feat', 0) + \
                         loss_dict.get('G_A_GAN', 0) + loss_dict.get('G_y_0_GAN', 0) + \
                         loss_dict.get('G_A_VGG', 0) + loss_dict.get('smooth_y_0_loss', 0) + \
                         loss_dict.get('G_y_0_VGG', 0) + loss_dict.get('smooth_y_0_loss', 0) + \
                         loss_dict.get('G_GAN_CG', 0) + loss_dict.get('G_KL', 0)
                self.optG.zero_grad()
                loss_G.backward()
                self.optG.step()

                loss = sum(loss_dict.values())
            elif self.netG.pattern == 'B':
                loss_dict, generator = self.netG.trainB(self.gt_image, synthetic=self.synthetic, mask=self.mask,
                                                        mask_synthetic=self.mask_synthetic,
                                                        inst=self.inst, infer=infer, optD=self.optD_y_0)
                loss_G = loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + \
                         loss_dict.get('G_GAN', 0) + loss_dict.get('smooth_loss', 0) + loss_dict.get('G_KL', 0)

                self.optG.zero_grad()
                loss_G.backward()
                self.optG.step()

                loss = sum(loss_dict.values())
            elif self.netG.pattern == 'MAP':
                loss_dict, generator = self.netG.trainMAP(self.gt_image, synthetic=self.synthetic, mask=self.mask,
                                                          mask_abs=self.mask_abs,
                                                          mask_synthetic=self.mask_synthetic,
                                                          inst=self.inst, infer=infer, optD_y_0=self.optD_y_0)
                loss_G = loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_GAN', 0) + \
                         loss_dict.get('G_featD', 0) + loss_dict.get('smooth_loss', 0) + loss_dict.get('L1', 0)

                self.optG.zero_grad()
                loss_G.backward()
                self.optG.step()

                loss = sum(loss_dict.values())
            elif self.netG.pattern == 'NP':

                loss_dict, generator = self.netG.trainNP(self.gt_image, synthetic=self.synthetic, mask=self.mask,
                                                         mask_abs=self.mask_abs,
                                                         mask_synthetic=self.mask_synthetic,
                                                         inst=self.inst, infer=infer)
                loss_G = loss_dict.get('Mse', 0)

                self.optG.zero_grad()
                loss_G.backward()
                self.optG.step()

                loss = sum(loss_dict.values())
            else:
                self.optG.zero_grad()
                loss, generator = self.netG(self.gt_image, synthetic=self.synthetic, mask=self.mask,
                                            mask_synthetic=self.mask_synthetic,
                                            inst=self.inst, infer=infer)
                loss.backward()
                self.optG.step()
            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if infer:
                save_path = os.path.join(self.opt['path']['base_dir'], self.netG.pattern)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                imgs_num = self.gt_image.shape[0]
                imgs = generator.clamp_(-1.0, 1.0).data
                imgs = (imgs + 1.) / 2.0
                try:
                    vutils.save_image(imgs, save_path + '/' + str(self.epoch) + '_' + str(
                        self.iter) + '.png', nrow=imgs_num, padding=0, normalize=True)
                except OSError as err:
                    print(err)
                if loss_dict:
                    print(", ".join(f"{key}: {str(value.item())[:6]}" for key, value in loss_dict.items()))
                else:
                    print(f'Loss{loss.item()}')
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            self.netG.reset_new_noise_schedule(phase='valid')
            for val_data in tqdm.tqdm(self.val_loader, leave=False, desc='sampling image',
                                      total=len(self.val_loader), position=0):  # 测试数据
                self.set_input(val_data)
                self.output, self.visuals = self.netG.restoration(y_0=self.gt_image, synthetic=self.synthetic,
                                                                  mask_synthetic=self.mask_synthetic,
                                                                  mask=self.mask, mask_abs=self.mask_abs,
                                                                  sample_num=self.sample_num, inst=self.inst)
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                self.writer.save_images(self.save_current_results())
            #############恢复训练采样##############################
            self.netG.set_new_noise_schedule(phase='train')
            #####################################################
        return self.val_metrics.result()

    def test(self):
        self.netG.eval()  # 将Unet设置为测试模式
        self.test_metrics.reset()  # 重置log信息
        with ((torch.no_grad())):
            for phase_data in tqdm.tqdm(self.phase_loader, position=0, desc='sampling image', leave=True):
                self.set_input(phase_data)  # 将数据同步到Palette对象中
                self.output, self.visuals = self.netG.restoration(y_0=self.gt_image,
                                                                  synthetic=self.synthetic,
                                                                  mask=self.mask,
                                                                  mask_abs=self.mask_abs,
                                                                  mask_synthetic=self.mask_synthetic,
                                                                  sample_num=self.sample_num, inst=self.inst)
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')  # 在writer写入信息
                self.writer.save_images(self.save_current_results())  # 保存当前过程和结果；self.save_current_results()获得过程和结果

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        netG_label = self.netG.__class__.__name__  # 获得模型的名字
        self.load_network(network=self.netG, network_label=netG_label, strict=True)  # 为模型加载参数

    # 保存模型参数
    def save_everything(self):
        """ load pretrained model and training state. """
        netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)  # 保存模型，保存优化器
        self.save_training_state()  # 保存优化器
        ###########删除过多的检查点#####################
        # 获取检查点目录
        directory = Path(self.opt['path']['checkpoint'])
        max_files = 20 if self.netG.pattern == 'MAP' else 80

        # 获取所有文件
        files = [f for f in directory.iterdir() if f.is_file()]

        # 如果文件数量超过限制，删除较早的文件
        if len(files) > max_files:
            # 按修改时间排序，较早的文件在前
            files.sort(key=lambda f: f.stat().st_mtime)
            # 删除多余的文件
            for file_to_delete in files[:len(files) - max_files]:
                file_to_delete.unlink()  # 删除文件
                print(f"Deleted: {file_to_delete.name}")  # 只打印文件名

        # 如果图片数量超过限制，删除较早的图片
        # 构建图像目录路径
        img_dir = Path(self.opt['path']['base_dir']) / self.netG.pattern
        if img_dir.exists():
            img_max_length = 1000  # 假设的最大文件数量
            # 获取所有文件并按修改时间排序
            img_path = sorted(img_dir.iterdir(), key=lambda f: f.stat().st_mtime)
            # 删除多余的文件
            if len(img_path) > img_max_length:
                for img_to_delete in img_path[:len(img_path) - img_max_length]:
                    img_to_delete.unlink()  # 删除文件
