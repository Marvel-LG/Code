import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt)
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    opt['world_size'] = 1
    main_worker(0, 1, opt)