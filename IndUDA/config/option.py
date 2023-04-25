import argparse
import os
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use',
                        default='CAN', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='exp', type=str)
    parser.add_argument('--inn_weight', dest='inn_weight',
                        help='inn_loss_weigh',
                        default='0.5', type=float)
    parser.add_argument('--inn_norm_weight', dest='inn_norm_weight',
                        help='inn_norm_loss_weigh',
                        default='0.5', type=float)
    parser.add_argument('--con_weight', dest='con_weight',
                        help='con_weigh',
                        default='1.0', type=float)
    parser.add_argument('--block_num', dest='block_num',
                        help='number of inn block',
                        default='3', type=int)
    parser.add_argument('--ft', dest='ft',
                        help='ft',
                        default='1.0', type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args