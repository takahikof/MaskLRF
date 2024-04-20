import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type = str,
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')
    parser.add_argument(
        '--use_tfboard',
        action='store_true',
        default=False,
        help = 'use TFBoard or not')
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)


    ######################################
    # additional commandline arguments
    # -1 indicates "not specified"
    parser.add_argument('--myarg_batch_size', type=int, default=-1 )
    parser.add_argument('--myarg_rot_train', type=str, default="-1" )
    parser.add_argument('--myarg_rot_test', type=str, default="-1" )
    parser.add_argument('--myarg_lrf_axis1', type=str, default="-1" )
    parser.add_argument('--myarg_lrf_axis2', type=str, default="-1" )
    parser.add_argument('--myarg_output_feature_type', type=str, default="-1" )
    parser.add_argument('--myarg_pod_gridsize', type=int, default=-1 )
    parser.add_argument('--myarg_trans_depth', type=int, default=-1 )
    parser.add_argument('--myarg_group_size', type=int, default=-1 )
    parser.add_argument('--myarg_num_group', type=int, default=-1 )
    parser.add_argument('--myarg_mask_ratio', type=float, default=-1 )
    parser.add_argument('--myarg_attn_mode', type=str, default="-1" )
    parser.add_argument('--myarg_attn_subsample_rate', type=float, default=-1 )
    parser.add_argument('--myarg_attn_target', type=str, default="-1" )

    parser.add_argument('--partseg', action='store_true', default=False)
    parser.add_argument('--domain_adaptation', action='store_true', default=False)
    ######################################

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # if args.test:
    #     args.exp_name = 'test_' + args.exp_name
    # if args.mode is not None:
    #     args.exp_name = args.exp_name + '_' +args.mode
    # args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    # args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    # args.log_name = Path(args.config).stem
    # create_experiment_dir(args)

    args.experiment_path = os.path.join('./experiments', args.exp_name )
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if( args.use_tfboard ):
        args.tfboard_path = os.path.join('./experiments', args.exp_name, 'TFBoard' )
        if not os.path.exists(args.tfboard_path):
            os.makedirs(args.tfboard_path)
            print('Create TFBoard path successfully at %s' % args.tfboard_path)

    args.log_name = "log"

    return args

# def create_experiment_dir(args):
#     if not os.path.exists(args.experiment_path):
#         os.makedirs(args.experiment_path)
#         print('Create experiment path successfully at %s' % args.experiment_path)
#     if not os.path.exists(args.tfboard_path):
#         os.makedirs(args.tfboard_path)
#         print('Create TFBoard path successfully at %s' % args.tfboard_path)
