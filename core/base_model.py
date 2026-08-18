import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn
from utils.ShowCount import Count_and_Value

import core.util as Util
import sys

CustomResult = collections.namedtuple('CustomResult', 'name result')


class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.opt = opt
        self.phase = opt['phase']
        self.set_device = partial(Util.set_device, rank=opt['global_rank'])

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = 0
        self.iter = 0

        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([], [])  # {"name":[], "result":[]}
        #############################
        self.valid_count = []
        self.train_count = []
        ###########################
        self.current_resum_root = None
        self.resume_A = opt['path']['resume_state_A']
        self.resume_B = opt['path']['resume_state_B']
        self.resume_MAP = opt['path']['resume_state_MAP']
        self.resume_D = opt['path']['resume_state_D']
        self.resume_NP = opt['path']['resume_state_NP']

    def train(self):
        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1
            train_log = self.train_step()

            self.train_count.append([self.epoch, train_log['train/mse_loss']])
            print(f'Epoch {self.epoch} Loss: {train_log["train/mse_loss"]}')
            Count_and_Value(self.train_count, f"{self.opt['path']['experiments_root']}/Count_train")

            ''' save logged informations into log dict '''
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard '''
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))

            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.opt['train']['val_epoch'] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    self.valid_count.append([self.epoch, val_log['val/mae']])
                    Count_and_Value(self.valid_count, f"{self.opt['path']['experiments_root']}/Count_valid")
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass

    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        if self.opt['global_rank'] != 0:
            return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module

        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        if self.opt['global_rank'] != 0:
            return

        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module

        def save_model_state(model, filename):
            state_dict = model.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            torch.save(state_dict, filename)


        if network.pattern == 'MAP':
            save_filename_MAP = '{}_{}_MAP.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_MAP)
            save_model_state(network.denoise_fn.Map, save_path)

            latest_map_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_MAP.pth')
            torch.save(network.denoise_fn.Map.state_dict(), latest_map_path)

            if network.discriminator_y_0 is not None:
                save_filename_D_y_0 = '{}_{}_D_y_0.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_y_0)
                save_model_state(network.discriminator_y_0, save_path)

                latest_d_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_y_0.pth')
                torch.save(network.discriminator_y_0.state_dict(), latest_d_path)

            # save_filename_B = '{}_{}_B.pth'.format(self.epoch, network_label)
            # save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_B)
            # save_model_state(network.denoise_fn.B, save_path)
            #
            # latest_B_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_B.pth')
            # torch.save(network.denoise_fn.B.state_dict(), latest_B_path)

        elif network.pattern == 'NP':
            save_filename_NP = '{}_{}_NP.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_NP)
            save_model_state(network.NPNet, save_path)

            latest_np_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_NP.pth')
            torch.save(network.NPNet.state_dict(), latest_np_path)

        elif network.pattern == 'A':

            save_filename_A = '{}_{}_A.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_A)
            save_model_state(network.denoise_fn, save_path)
            latest_a_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_A.pth')
            torch.save(network.denoise_fn.state_dict(), latest_a_path)

            save_filename_Sup = '{}_{}_Sup.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_Sup)
            save_model_state(network.SupCon, save_path)
            latest_sup_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_Sup.pth')
            torch.save(network.SupCon.state_dict(), latest_sup_path)

            save_filename_D_syn = '{}_{}_D_syn.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_syn)
            save_model_state(network.discriminator_syn, save_path)
            latest_d_syn_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_syn.pth')
            torch.save(network.discriminator_syn.state_dict(), latest_d_syn_path)

            if network.discriminator_y_0 is not None:
                save_filename_D_y_0 = '{}_{}_D_y_0.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_y_0)
                save_model_state(network.discriminator_y_0, save_path)
                latest_d_y_0_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_y_0.pth')
                torch.save(network.discriminator_y_0.state_dict(), latest_d_y_0_path)

            save_filename_D_feat = '{}_{}_D_feat.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_feat)
            save_model_state(network.discriminator_feat, save_path)
            latest_d_feat_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_feat.pth')
            torch.save(network.discriminator_feat.state_dict(), latest_d_feat_path)

            if network.discriminator_contra_gan is not None:
                save_filename_D_contral_gan = '{}_{}_D_cg.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_contral_gan)
                save_model_state(network.discriminator_contra_gan, save_path)
                latest_d_contral_gan_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_cg.pth')
                torch.save(network.discriminator_contra_gan.state_dict(), latest_d_contral_gan_path)
        elif network.pattern == 'AandMAP':

            save_filename_A = '{}_{}_A.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_A)
            save_model_state(network.denoise_fn.A, save_path)
            latest_a_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_A.pth')
            torch.save(network.denoise_fn.A.state_dict(), latest_a_path)

            save_filename_Sup = '{}_{}_Sup.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_Sup)
            save_model_state(network.SupCon, save_path)
            latest_sup_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_Sup.pth')
            torch.save(network.SupCon.state_dict(), latest_sup_path)

            save_filename_D_syn = '{}_{}_D_syn.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_syn)
            save_model_state(network.discriminator_syn, save_path)
            latest_d_syn_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_syn.pth')
            torch.save(network.discriminator_syn.state_dict(), latest_d_syn_path)
            if network.discriminator_y_0 is not None:
                save_filename_D_y_0 = '{}_{}_D_y_0.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_y_0)
                save_model_state(network.discriminator_y_0, save_path)
                latest_d_y_0_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_y_0.pth')
                torch.save(network.discriminator_y_0.state_dict(), latest_d_y_0_path)

            save_filename_D_feat = '{}_{}_D_feat.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_feat)
            save_model_state(network.discriminator_feat, save_path)
            latest_d_feat_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_feat.pth')
            torch.save(network.discriminator_feat.state_dict(), latest_d_feat_path)

            save_filename_MAP = '{}_{}_MAP.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_MAP)
            save_model_state(network.denoise_fn.Map, save_path)

            latest_map_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_MAP.pth')
            torch.save(network.denoise_fn.Map.state_dict(), latest_map_path)

            if network.discriminator_contra_gan is not None:
                save_filename_D_contral_gan = '{}_{}_D_cg.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_contral_gan)
                save_model_state(network.discriminator_contra_gan, save_path)
                latest_d_contral_gan_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_cg.pth')
                torch.save(network.discriminator_contra_gan.state_dict(), latest_d_contral_gan_path)

        elif network.pattern == 'B':
            save_filename_B = '{}_{}_B.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_B)
            save_model_state(network.denoise_fn, save_path)

            latest_b_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_B.pth')
            torch.save(network.denoise_fn.state_dict(), latest_b_path)
            if network.discriminator_y_0 is not None:
                save_filename_D_y_0 = '{}_{}_D_y_0.pth'.format(self.epoch, network_label)
                save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_D_y_0)
                save_model_state(network.discriminator_y_0, save_path)

                latest_d_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D_y_0.pth')
                torch.save(network.discriminator_y_0.state_dict(), latest_d_path)

        elif network.pattern == 'D':
            save_filename_P = '{}_{}_D.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], save_filename_P)
            save_model_state(network.denoise_fn, save_path)

            latest_d_path = os.path.join(self.opt['path']['checkpoint'], f'latest_{network_label}_D.pth')
            torch.save(network.denoise_fn.state_dict(), latest_d_path)


        iterpath = os.path.join(self.opt['path']['checkpoint'], 'iter.txt')
        with open(iterpath, 'w') as file:
            file.write(f"{self.epoch}\n")
            file.write(f"{self.iter}\n")

    def load_model(self, save_path, model):
        if os.path.exists(save_path):
            try:
                model.load_state_dict(torch.load(save_path, weights_only=True))
                print("Pretrained network %s loaded successfully!" % save_path)
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = model.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model.load_state_dict(pretrained_dict)
                    print(
                        "Pretrained network %s has excessive layers; Only loading layers that are used"
                        % save_path
                    )
                except:
                    print(
                        "Pretrained network %s has fewer layers; The following are not initialized:"
                        % save_path
                    )
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split(".")[0])

                    print(sorted(not_initialized))
                    model.load_state_dict(model_dict)
        else:
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(save_path))
            print('Pretrained model in [{:s}] is not existed, Skip it'.format(save_path))

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state_A'] is None and self.opt['path']['resume_state_B'] is None and \
                self.opt['path']['resume_state_MAP'] is None:
            return
        self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        if network.pattern == 'MAP':
            self.current_resum_root = self.opt['path']['resume_state_MAP'][
                                      : self.opt['path']['resume_state_MAP'].rfind('/')]
            model_path_A = os.path.join(self.resume_A, "latest_Network_A.pth")
            # model_path_B = os.path.join(self.resume_MAP, "latest_Network_B.pth")
            model_path_Map = os.path.join(self.resume_MAP, "latest_Network_MAP.pth")
            self.load_model(model_path_A, network.denoise_fn.A)
            # self.load_model(model_path_B, network.denoise_fn.B)
            self.load_model(model_path_Map, network.denoise_fn.Map)

            if network.discriminator_y_0 is not None:
                model_path_D_y_0 = os.path.join(self.current_resum_root, "latest_Network_D_y_0.pth")
                self.load_model(model_path_D_y_0, network.discriminator_y_0)

        elif network.pattern == 'AandMAP':
            self.current_resum_root = self.opt['path']['resume_state_AandMAP'][
                                      : self.opt['path']['resume_state_AandMAP'].rfind('/')]
            model_path_A = os.path.join(self.current_resum_root, "latest_Network_A.pth")
            model_path_Sup = os.path.join(self.current_resum_root, "latest_Network_Sup.pth")
            model_path_D_syn = os.path.join(self.current_resum_root, "latest_Network_D_syn.pth")

            model_path_D_feat = os.path.join(self.current_resum_root, "latest_Network_D_feat.pth")
            model_path_Map = os.path.join(self.current_resum_root, "latest_Network_MAP.pth")

            self.load_model(model_path_A, network.denoise_fn.A)
            self.load_model(model_path_Sup, network.SupCon)
            self.load_model(model_path_D_syn, network.discriminator_syn)
            self.load_model(model_path_D_feat, network.discriminator_feat)
            self.load_model(model_path_Map, network.denoise_fn.Map)

            if network.discriminator_y_0 is not None:
                model_path_D_y_0 = os.path.join(self.current_resum_root, "latest_Network_D_y_0.pth")
                self.load_model(model_path_D_y_0, network.discriminator_y_0)


            if network.discriminator_contra_gan is not None:
                model_path_D_contral_gan = os.path.join(self.current_resum_root, "latest_Network_D_cg.pth")
                self.load_model(model_path_D_contral_gan, network.discriminator_contra_gan)
        elif network.pattern == 'NP':
            self.current_resum_root = self.opt['path']['resume_state_NP'][
                                      : self.opt['path']['resume_state_NP'].rfind('/')]
            model_path_A = os.path.join(self.resume_A, "latest_Network_A.pth")
            model_path_B = os.path.join(self.resume_B, "latest_Network_B.pth")
            model_path_Map = os.path.join(self.resume_MAP, "latest_Network_MAP.pth")
            model_path_NP = os.path.join(self.resume_NP, "latest_Network_NP.pth")
            self.load_model(model_path_A, network.denoise_fn.A)
            self.load_model(model_path_B, network.denoise_fn.B)
            self.load_model(model_path_Map, network.denoise_fn.Map)
            self.load_model(model_path_NP, network.NPNet)
        elif network.pattern == 'A':
            self.current_resum_root = self.opt['path']['resume_state_A'][
                                      : self.opt['path']['resume_state_A'].rfind('/')]
            model_path_A = os.path.join(self.current_resum_root, "latest_Network_A.pth")
            model_path_Sup = os.path.join(self.resume_A, "latest_Network_Sup.pth")
            model_path_D_syn = os.path.join(self.current_resum_root, "latest_Network_D_syn.pth")
            model_path_D_feat = os.path.join(self.current_resum_root, "latest_Network_D_feat.pth")

            self.load_model(model_path_A, network.denoise_fn)
            self.load_model(model_path_Sup, network.SupCon)
            self.load_model(model_path_D_syn, network.discriminator_syn)
            self.load_model(model_path_D_feat, network.discriminator_feat)

            if network.discriminator_y_0 is not None:
                model_path_D_y_0 = os.path.join(self.current_resum_root, "latest_Network_D_y_0.pth")
                self.load_model(model_path_D_y_0, network.discriminator_y_0)

            if network.discriminator_contra_gan is not None:
                model_path_D_contral_gan = os.path.join(self.current_resum_root, "latest_Network_D_cg.pth")
                self.load_model(model_path_D_contral_gan, network.discriminator_contra_gan)
        elif network.pattern == 'B':
            self.current_resum_root = self.opt['path']['resume_state_B'][
                                      : self.opt['path']['resume_state_B'].rfind('/')]
            model_path_B = os.path.join(self.current_resum_root, "latest_Network_B.pth")
            self.load_model(model_path_B, network.denoise_fn)

            if network.discriminator_y_0 is not None:
                model_path_D_y_0 = os.path.join(self.current_resum_root, "latest_Network_D_y_0.pth")
                self.load_model(model_path_D_y_0, network.discriminator_y_0)
        elif network.pattern == 'D':
            self.current_resum_root = self.opt['path']['resume_state_D'][
                                      : self.opt['path']['resume_state_D'].rfind('/')]
            model_path_D = os.path.join(self.current_resum_root, "latest_Network_D.pth")
            self.load_model(model_path_D, network.denoise_fn)
        elif network.pattern == 'R':
            self.current_resum_root = self.opt['path']['resume_state_MAP'][
                                      : self.opt['path']['resume_state_MAP'].rfind('/')]
            model_path_A = os.path.join(self.resume_A, "latest_Network_A.pth")
            model_path_B = os.path.join(self.resume_B, "latest_Network_B.pth")
            model_path_Map = os.path.join(self.resume_MAP, "latest_Network_MAP.pth")
            self.load_model(model_path_A, network.denoise_fn.A)
            self.load_model(model_path_B, network.denoise_fn.B)
            self.load_model(model_path_Map, network.denoise_fn.Map)
        ###################################################

    def save_optimizer(self, optimizer, savename):
        save_path = os.path.join(self.opt['path']['checkpoint'], savename)
        torch.save(optimizer.state_dict(), save_path)

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        if self.opt['global_rank'] != 0:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers,
                                                                list), 'optimizers and schedulers must be a list.'

        if self.netG.pattern == 'MAP':
            map_save_filename = '{}_MAP.state'.format(self.epoch)
            latest_map_save_filename = 'latest_MAP.state'
            self.save_optimizer(self.optG, map_save_filename)
            self.save_optimizer(self.optG, latest_map_save_filename)
            if self.optD_y_0 is not None:
                map_d_y_0_save_filename = '{}_D_y_0.state'.format(self.epoch)
                latest_map_d_y_0_save_filename = 'latest_D_y_0.state'
                self.save_optimizer(self.optD_y_0, map_d_y_0_save_filename)
                self.save_optimizer(self.optD_y_0, latest_map_d_y_0_save_filename)
        elif self.netG.pattern == 'AandMAP':
            aandmap_save_filename = '{}_AandMAP.state'.format(self.epoch)
            latest_aandmap_save_filename = 'latest_AandMAP.state'
            self.save_optimizer(self.optG, aandmap_save_filename)
            self.save_optimizer(self.optG, latest_aandmap_save_filename)

            a_d_syn_save_filename = '{}_D_syn.state'.format(self.epoch)
            latest_a_d_syn_save_filename = 'latest_D_syn.state'
            self.save_optimizer(self.optD_syn, a_d_syn_save_filename)
            self.save_optimizer(self.optD_syn, latest_a_d_syn_save_filename)
            if self.optD_y_0 is not None:
                a_d_y_0_save_filename = '{}_D_y_0.state'.format(self.epoch)
                latest_a_d_y_0_save_filename = 'latest_D_y_0.state'
                self.save_optimizer(self.optD_y_0, a_d_y_0_save_filename)
                self.save_optimizer(self.optD_y_0, latest_a_d_y_0_save_filename)

            a_d_feat_save_filename = '{}_D_feat.state'.format(self.epoch)
            latest_a_d_feat_save_filename = 'latest_D_feat.state'
            self.save_optimizer(self.optD_feat, a_d_feat_save_filename)
            self.save_optimizer(self.optD_feat, latest_a_d_feat_save_filename)

            if self.optD_cg is not None:
                a_d_contral_gan_save_filename = '{}_D_cg.state'.format(self.epoch)
                latest_a_d_contral_gan_save_filename = 'latest_D_cg.state'
                self.save_optimizer(self.optD_cg, a_d_contral_gan_save_filename)
                self.save_optimizer(self.optD_cg, latest_a_d_contral_gan_save_filename)

        elif self.netG.pattern == 'NP':
            np_save_filename = '{}_NP.state'.format(self.epoch)
            latest_np_save_filename = 'latest_NP.state'
            self.save_optimizer(self.optG, np_save_filename)
            self.save_optimizer(self.optG, latest_np_save_filename)

        elif self.netG.pattern == 'A':
            a_save_filename = '{}_A.state'.format(self.epoch)
            latest_a_save_filename = 'latest_A.state'
            self.save_optimizer(self.optG, a_save_filename)
            self.save_optimizer(self.optG, latest_a_save_filename)

            a_d_syn_save_filename = '{}_D_syn.state'.format(self.epoch)
            latest_a_d_syn_save_filename = 'latest_D_syn.state'
            self.save_optimizer(self.optD_syn, a_d_syn_save_filename)
            self.save_optimizer(self.optD_syn, latest_a_d_syn_save_filename)
            if self.optD_y_0 is not None:
                a_d_y_0_save_filename = '{}_D_y_0.state'.format(self.epoch)
                latest_a_d_y_0_save_filename = 'latest_D_y_0.state'
                self.save_optimizer(self.optD_y_0, a_d_y_0_save_filename)
                self.save_optimizer(self.optD_y_0, latest_a_d_y_0_save_filename)

            a_d_feat_save_filename = '{}_D_feat.state'.format(self.epoch)
            latest_a_d_feat_save_filename = 'latest_D_feat.state'
            self.save_optimizer(self.optD_feat, a_d_feat_save_filename)
            self.save_optimizer(self.optD_feat, latest_a_d_feat_save_filename)

            if self.optD_cg is not None:
                a_d_contral_gan_save_filename = '{}_D_cg.state'.format(self.epoch)
                latest_a_d_contral_gan_save_filename = 'latest_D_cg.state'
                self.save_optimizer(self.optD_cg, a_d_contral_gan_save_filename)
                self.save_optimizer(self.optD_cg, latest_a_d_contral_gan_save_filename)


        elif self.netG.pattern == 'B':
            b_save_filename = '{}_B.state'.format(self.epoch)
            latest_b_save_filename = 'latest_B.state'
            self.save_optimizer(self.optG, b_save_filename)
            self.save_optimizer(self.optG, latest_b_save_filename)
            if self.optD_y_0 is not None:
                b_d_save_filename = '{}_D_y_0.state'.format(self.epoch)
                latest_b_d_save_filename = 'latest_D_y_0.state'
                self.save_optimizer(self.optD_y_0, b_d_save_filename)
                self.save_optimizer(self.optD_y_0, latest_b_d_save_filename)

        elif self.netG.pattern == 'D':
            d_save_filename = '{}_D.state'.format(self.epoch)
            latest_d_save_filename = 'latest_D.state'
            self.save_optimizer(self.optG, d_save_filename)
            self.save_optimizer(self.optG, latest_d_save_filename)

    def load_optimizer(self, optimizer, load_path):
        if not os.path.isfile(load_path):
            print("Optimizer model in %s not exists yet!" % load_path)
        else:
            optimizer.load_state_dict(torch.load(load_path, weights_only=True))
            print("Optimizer model in %s loaded successfully!" % load_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase != 'train':
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers,
                                                                list), 'optimizers and schedulers must be a list.'

        if self.netG.pattern == 'MAP':
            map_state_path = os.path.join(self.opt['path']['resume_state_MAP'], 'latest_MAP.state')
            self.load_optimizer(self.optG, map_state_path)
            if self.optD_y_0 is not None:
                map_d_y_0_state_path = os.path.join(self.opt['path']['resume_state_MAP'], 'latest_D_y_0.state')
                self.load_optimizer(self.optD_y_0, map_d_y_0_state_path)
            else:
                self.optD_y_0 = None
        elif self.netG.pattern == 'AandMAP':
            a_d_syn_state_path = os.path.join(self.opt['path']['resume_state_AandMAP'], "latest_D_syn.state")
            self.load_optimizer(self.optD_syn, a_d_syn_state_path)

            a_d_feat_state_path = os.path.join(self.opt['path']['resume_state_AandMAP'], "latest_D_feat.state")
            self.load_optimizer(self.optD_feat, a_d_feat_state_path)
            if self.optD_y_0 is not None:
                a_d_y_0_state_path = os.path.join(self.opt['path']['resume_state_AandMAP'], 'latest_D_y_0.state')
                self.load_optimizer(self.optD_y_0, a_d_y_0_state_path)
            else:
                self.optD_y_0 = None

            map_state_path = os.path.join(self.opt['path']['resume_state_AandMAP'], 'latest_AandMAP.state')
            self.load_optimizer(self.optG, map_state_path)

            if self.optD_cg is not None:
                a_d_contral_gan_state_path = os.path.join(self.opt['path']['resume_state_AandMAP'], 'latest_D_cg.state')
                self.load_optimizer(self.optD_cg, a_d_contral_gan_state_path)
        elif self.netG.pattern == 'NP':
            np_state_path = os.path.join(self.opt['path']['resume_state_NP'], 'latest_NP.state')
            self.load_optimizer(self.optG, np_state_path)

        elif self.netG.pattern == 'A':
            a_state_path = os.path.join(self.opt['path']['resume_state_A'], "latest_A.state")
            self.load_optimizer(self.optG, a_state_path)

            a_d_syn_state_path = os.path.join(self.opt['path']['resume_state_A'], "latest_D_syn.state")
            self.load_optimizer(self.optD_syn, a_d_syn_state_path)

            a_d_feat_state_path = os.path.join(self.opt['path']['resume_state_A'], "latest_D_feat.state")
            self.load_optimizer(self.optD_feat, a_d_feat_state_path)
            if self.optD_y_0 is not None:
                a_d_y_0_state_path = os.path.join(self.opt['path']['resume_state_A'], 'latest_D_y_0.state')
                self.load_optimizer(self.optD_y_0, a_d_y_0_state_path)
            else:
                self.optD_y_0 = None
            if self.optD_cg is not None:
                a_d_contral_gan_state_path = os.path.join(self.opt['path']['resume_state_A'], 'latest_D_cg.state')
                self.load_optimizer(self.optD_cg, a_d_contral_gan_state_path)
            else:
                self.optD_cg = None
        elif self.netG.pattern == 'B':
            b_state_path = os.path.join(self.opt['path']['resume_state_B'], 'latest_B.state')
            self.load_optimizer(self.optG, b_state_path)
            if self.optD_y_0 is not None:
                b_d_y_0_state_path = os.path.join(self.opt['path']['resume_state_B'], 'latest_D_y_0.state')
                self.load_optimizer(self.optD_y_0, b_d_y_0_state_path)

        elif self.netG.pattern == 'D':
            d_state_path = os.path.join(self.opt['path']['resume_state_D'], 'latest_D.state')
            self.load_optimizer(self.optG, d_state_path)
        else:
            raise Exception(f'Not pattern parameter：{self.netG.pattern}')

        print("---------- Optimizers initialized -------------")
        iterpath = os.path.join(self.current_resum_root, 'iter.txt')
        print("epoch and iter path: %s" % iterpath)
        if os.path.isfile(iterpath):
            with open(iterpath, 'r') as file:
                lines = file.readlines()
                epoch = int(lines[0].strip())
                iteration = int(lines[1].strip())
        else:
            epoch = 0
            iteration = 0
            with open(iterpath, 'w') as file:
                file.write(f"{epoch}\n")
                file.write(f"{iteration}\n")
        self.epoch = epoch
        self.iter = iteration

    def load_everything(self):
        pass

    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')
