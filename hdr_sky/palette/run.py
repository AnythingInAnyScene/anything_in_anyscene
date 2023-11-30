import argparse
import os, glob
import warnings
import torch
import torch.multiprocessing as mp
import cv2
import numpy as np
import pkg_resources

from anything_in_anyscene.hdr_sky.palette.core.logger import VisualWriter, InfoLogger
import anything_in_anyscene.hdr_sky.palette.core.praser as Praser
import anything_in_anyscene.hdr_sky.palette.core.util as Util
from anything_in_anyscene.hdr_sky.palette.data import define_dataloader
from anything_in_anyscene.hdr_sky.palette.models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
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

def setup_input_image(input_path, save_path):
    img_list = glob.glob(os.path.join(input_path, "*g"))
    input_img = cv2.imread(img_list[0])
    h, w = input_img.shape[:2] 
    output_img = np.zeros_like(input_img)
    delta = 10
    input_img = cv2.resize(input_img[:h//3, :, :], (w//3 + 10, h//3 + 10))
    output_img[h//3*2-10:, w//3-5:w//3*2+5, :] = input_img
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path, "sky.png"), output_img)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-t', '--test_path', type=str, default='/data')
    parser.add_argument('-s', '--save_path', type=str, default='/all/palette_tmp')

    ''' parser configs '''
    args = parser.parse_args()
    args.config = pkg_resources.resource_filename("anything_in_anyscene.hdr_sky", "palette/config/customize_laval_sky.json")
    opt = Praser.parse(args)
    opt["datasets"]["test"]["which_dataset"]["args"]["data_root"] = args.save_path
    setup_input_image(args.test_path, opt["datasets"]["test"]["which_dataset"]["args"]["data_root"])
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
        # main_worker([0,1,2,3], 1, opt)