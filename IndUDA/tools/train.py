import argparse
import os
import sys
import torch
import datetime
now=datetime.datetime.now()
time = now.strftime("%Y-%m-%d")
base_path = os.getcwd()
sys.path.append(base_path)
from torch.backends import cudnn
from time import strftime
from config.config import cfg_from_file, cfg_from_list
from config.option import parse_args
from prepare_data import *
import sys
import pprint
from model.model import *
from model.Inv_Model import InvNet, subnet_constructor
from logger import setup_logger
from solver.INN_solver import INNSolver as Solver

backbones = [resnet]


def train(args):
    bn_domain_map = {}
    # method-specific setting

    
    dataloaders = prepare_data_CAN()
    num_domains_bn = 2

    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        model_state_dict = resume_dict['model_state_dict']
        fx_pretrained = False
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = param_dict['weights']
        bn_domain_map = param_dict['bn_domain_map']
        fx_pretrained = False

    net = danet(num_classes=cfg.DATASET.NUM_CLASSES,
                 state_dict=model_state_dict,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
                 frozen=[cfg.TRAIN.STOP_GRAD],
                 fx_pretrained=fx_pretrained,
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                 num_domains_bn=num_domains_bn,
                 block_num=cfg.HYPER.BLOCK_NUM)
    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # initialize solver
    train_solver = Solver(net, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict)

    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume
    if args.weights is not None:
        cfg.MODEL = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, str(cfg.DATASET.SOURCE_NAME[0]+'_'+ cfg.DATASET.TARGET_NAME[0]),
                 'bn' + str(cfg.HYPER.BLOCK_NUM) + '_ft' + str(cfg.CLUSTERING.FILTERING_THRESHOLD))
    # cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, str(cfg.DATASET.SOURCE_NAME[0]+'_'+ cfg.DATASET.TARGET_NAME[0]),
    #              'max')

    if not os.path.exists(cfg.SAVE_DIR):
        print(cfg.SAVE_DIR)
        os.makedirs(cfg.SAVE_DIR)
    setup_logger(cfg.SAVE_DIR)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)
    train(args)