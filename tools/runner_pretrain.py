import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tools import builder
from utils import misc, dist_utils, retrieval
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

# from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

rotateSO3 = transforms.Compose( [ data_transforms.PointcloudRotateSO3() ] )

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAnisotropic(scale_low=0.8, scale_high=1.2),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    train_sampler, train_dataloader, num_train_samples = builder.dataset_builder(args, config.dataset.train)
    _, test_dataloader, _ = builder.dataset_builder(args, config.dataset.val)
    _, train_dataloader_inference, _ = builder.dataset_builder(args, config.dataset.train_inference)
    # (_, extra_train_dataloader, _)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    # build model
    config.model.num_train_samples = num_train_samples
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['RecLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, instance_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ShapeNet_withnormal':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints

            if( config.rot_train == "so3" ):
                points = rotateSO3( points )

            points = train_transforms( points )

            overall_loss = base_model( points )

            overall_loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([overall_loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            base_model.eval()  # set model to evaluation mode
            metrics = validate(base_model, test_dataloader, epoch, args, config, logger=logger)
            base_model.train()  # set model to training mode

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        # if epoch % 25 ==0 and epoch >=250:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
        #                             logger=logger)

def extract_global_features(base_model, test_dataloader, args, config, logger = None):

    test_features = []
    test_labels = []
    test_instance_ids = []

    # for ShapeNet55
    taxonomyID_to_categoryID = {'03759954': 0, '04330267': 1, '03261776': 2,
    '03513137': 3, '04460130': 4, '04468005': 5, '03761084': 6, '04004475': 7,
    '03938244': 8, '02992529': 9, '02808440': 10, '04530566': 11, '02871439': 12,
    '03593526': 13, '04554684': 14, '03467517': 15, '04401088': 16, '02954340': 17,
    '02933112': 18, '03642806': 19, '02924116': 20, '02946921': 21, '02818832': 22,
    '04256520': 23, '04379243': 24, '02747177': 25, '03046257': 26, '04225987': 27,
    '03797390': 28, '03001627': 29, '03691459': 30, '02942699': 31, '03636649': 32,
    '02691156': 33, '02773838': 34, '02958343': 35, '02880940': 36, '04099429': 37,
    '02828884': 38, '04090263': 39, '03211117': 40, '02876657': 41, '03710193': 42,
    '03790512': 43, '03325088': 44, '02801938': 45, '03337140': 46, '02843684': 47,
    '03624134': 48, '03991062': 49, '03948459': 50, '03928116': 51, '03207941': 52,
    '04074963': 53, '03085013': 54}

    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, instance_id, data) in enumerate(test_dataloader):

            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                label = []
                for i in range( len( taxonomy_ids ) ) :
                    label.append( taxonomyID_to_categoryID[ taxonomy_ids[ i ] ] )
                label = np.array( label )
                label = torch.from_numpy( label.astype(np.int32)).clone()
            elif dataset_name == 'ShapeNet_withnormal':
                points = data.cuda()
                label = taxonomy_ids.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                label = data[1].cuda()
                points = misc.fps(points, npoints)

            assert points.size(1) == npoints

            if( config.rot_test == "so3" ):
                points = rotateSO3( points )

            feature = base_model(points, extract_global_features=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_labels.append(target.detach())
            test_instance_ids.append(instance_id.detach())

        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        test_instance_ids = torch.cat(test_instance_ids, dim=0)

    return test_features, test_labels, test_instance_ids

def validate(base_model, test_dataloader, epoch, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    test_features, test_labels, _ = extract_global_features(base_model, test_dataloader, args, config, logger=logger)
    _, _, macro_map, _ = retrieval.retrieval( test_features.data.cpu().numpy(), test_labels.data.cpu().numpy() )
    print_log('[Validation] EPOCH: %d  macroMAP = %.4f' % (epoch,macro_map), logger=logger)

    return Acc_Metric(macro_map)


def test_net():
    pass
