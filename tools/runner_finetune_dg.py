import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

rotateSO3 = transforms.Compose( [ data_transforms.PointcloudRotateSO3() ] )

crop_transforms = transforms.Compose(
    [
         data_transforms.PointcloudRandomCrop(),
    ]
)

mixup_transforms = transforms.Compose(
    [
         data_transforms.PointcloudMixup(),
    ]
)

dataaug_transforms = transforms.Compose(
    [
         data_transforms.PointcloudScaleAnisotropic(scale_low=0.8, scale_high=1.2),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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
    train_sampler, trainS_dataloader, _ = builder.dataset_builder(args, config.dataset.train_source)
    _, trainT_dataloader, _ = builder.dataset_builder(args, config.dataset.train_target)
    _, test_dataloader, _ = builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_epoch = 0
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
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
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(trainS_dataloader)

        npoints = config.npoints

        trainT_iter = iter( trainT_dataloader )
        count_iter = 0

        for idx, (taxonomy_ids, model_ids, dataS ) in enumerate(trainS_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            # for i in range( data[0].shape[0] ):
            #     out_filepath = "../data_downloaded/xyz_samples/ModelNetFewshot/" + str( i ) + ".xyz"
            #     np.savetxt( out_filepath, data[0][i] )
            # quit()

            pointsS = dataS[0].cuda()
            labelS = dataS[1].cuda()

            # if npoints == 1024:
            #     point_all = 1200
            # elif npoints == 2048:
            #     point_all = 2400
            # elif npoints == 4096:
            #     point_all = 4800
            # elif npoints == 8192:
            #     point_all = 8192
            # else:
            #     raise NotImplementedError()
            #
            # if pointsS.size(1) < point_all:
            #     point_all = pointsS.size(1)
            #
            # fps_idx = pointnet2_utils.furthest_point_sample(pointsS, point_all)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            # pointsS = pointnet2_utils.gather_operation(pointsS.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

            if( config.rot_train == "so3" ):
                pointsS = rotateSO3( pointsS )

            #########################################
            pointsS = crop_transforms( pointsS )
            pointsS, lambdasS, idxsS = mixup_transforms( pointsS )
            labelS2 = labelS[ idxsS ]
            pointsS = dataaug_transforms( pointsS )
            ret = base_model( pointsS, cls=True )
            loss_cls, acc = base_model.module.get_loss_acc_mixup( ret, labelS, labelS2, lambdasS )

            if( count_iter >= len( trainT_iter ) ):
                trainT_iter = iter( trainT_dataloader )
                count_iter = 0
            _, _, dataT = next( trainT_iter )
            count_iter += 1
            pointsT = dataT[0].cuda()
            # labelT = dataT[1].cuda()
            if( config.rot_train == "so3" ):
                pointsT = rotateSO3( pointsT )
            pointsT = dataaug_transforms( pointsT )
            loss_rec = base_model( pointsT, cls=False )

            _loss = loss_cls + loss_rec

            # start_time = time.time()
            _loss.backward()
            # print( "backward: " + str( time.time() - start_time ) )

            # start_time = time.time()
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            # print( "step: " + str( time.time() - start_time ) )

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                # losses.update([loss.item(), acc.item()])
                losses.update([loss_cls.item(), loss_rec.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                best_epoch = epoch
                # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            # if args.vote:
            #     if metrics.acc > 90.5 :
            #         metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            #         if metrics_vote.better_than(best_metrics_vote):
            #             best_metrics_vote = metrics_vote
            #             print_log(
            #                 "****************************************************************************************",
            #                 logger=logger)
            #             # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

            print_log('Current best acc = %.4f at %d th epoch' % ( best_metrics.acc, best_epoch ), logger = logger)

        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)

        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            if( config.rot_test == "so3" ):
                points = rotateSO3( points )

            logits = base_model(points, cls=True)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


# def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
#     print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
#     base_model.eval()  # set model to eval mode
#
#     test_pred  = []
#     test_label = []
#     npoints = config.npoints
#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             points_raw = data[0].cuda()
#             label = data[1].cuda()
#
#             local_pred = []
#
#             B, P, C = points_raw.shape
#
#             for kk in range(times):
#                 points = misc.fps( points_raw, npoints ) # P -> npoints
#
#                 if( config.rot_test == "so3" ):
#                     points = rotateSO3( points )
#
#                 points = test_transforms(points)
#
#                 logits = base_model(points)
#                 target = label.view(-1)
#
#                 local_pred.append(logits.detach().unsqueeze(0))
#
#                 # randomly shuffle points so that different 3d point is used as an initial point for FPS
#                 rand_idx = torch.randperm( P ).to(torch.int64).cuda()
#                 rand_idx = rand_idx.reshape( 1, P, 1 ).repeat( B, 1, C )
#                 points_raw = torch.gather( points_raw, dim=1, index=rand_idx )
#
#             pred = torch.cat(local_pred, dim=0).mean(0)
#             _, pred_choice = torch.max(pred, -1)
#
#             test_pred.append(pred_choice)
#             test_label.append(target.detach())
#
#         test_pred = torch.cat(test_pred, dim=0)
#         test_label = torch.cat(test_label, dim=0)
#
#         if args.distributed:
#             test_pred = dist_utils.gather_tensor(test_pred, args)
#             test_label = dist_utils.gather_tensor(test_label, args)
#
#         acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
#         print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)
#
#         if args.distributed:
#             torch.cuda.synchronize()
#
#     # Add testing results to TensorBoard
#     if val_writer is not None:
#         val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
#
#     return Acc_Metric(acc)

# def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
#     print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
#     base_model.eval()  # set model to eval mode
#
#     test_pred  = []
#     test_label = []
#     npoints = config.npoints
#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             points_raw = data[0].cuda()
#             label = data[1].cuda()
#             if npoints == 1024:
#                 point_all = 1200
#             elif npoints == 4096:
#                 point_all = 4800
#             elif npoints == 8192:
#                 point_all = 8192
#             else:
#                 raise NotImplementedError()
#
#             if points_raw.size(1) < point_all:
#                 point_all = points_raw.size(1)
#
#             fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
#             local_pred = []
#
#             for kk in range(times):
#                 fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
#                 points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
#                                                         fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
#
#                 # points = normal_estimation( points )
#                 points = test_transforms(points)
#
#                 logits = base_model(points)
#                 target = label.view(-1)
#
#                 local_pred.append(logits.detach().unsqueeze(0))
#
#             pred = torch.cat(local_pred, dim=0).mean(0)
#             _, pred_choice = torch.max(pred, -1)
#
#
#             test_pred.append(pred_choice)
#             test_label.append(target.detach())
#
#         test_pred = torch.cat(test_pred, dim=0)
#         test_label = torch.cat(test_label, dim=0)
#
#         if args.distributed:
#             test_pred = dist_utils.gather_tensor(test_pred, args)
#             test_label = dist_utils.gather_tensor(test_label, args)
#
#         acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
#         print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)
#
#         if args.distributed:
#             torch.cuda.synchronize()
#
#     # Add testing results to TensorBoard
#     if val_writer is not None:
#         val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
#
#     return Acc_Metric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

# def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
#
#     base_model.eval()  # set model to eval mode
#
#     test_pred  = []
#     test_label = []
#     npoints = config.npoints
#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             points_raw = data[0].cuda()
#             label = data[1].cuda()
#             if npoints == 1024:
#                 point_all = 1200
#             elif npoints == 4096:
#                 point_all = 4800
#             elif npoints == 8192:
#                 point_all = 8192
#             else:
#                 raise NotImplementedError()
#
#             if points_raw.size(1) < point_all:
#                 point_all = points_raw.size(1)
#
#             fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
#             local_pred = []
#
#             for kk in range(times):
#                 fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
#                 points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
#                                                         fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
#
#                 points = test_transforms(points)
#
#                 logits = base_model(points)
#                 target = label.view(-1)
#
#                 local_pred.append(logits.detach().unsqueeze(0))
#
#             pred = torch.cat(local_pred, dim=0).mean(0)
#             _, pred_choice = torch.max(pred, -1)
#
#
#             test_pred.append(pred_choice)
#             test_label.append(target.detach())
#
#         test_pred = torch.cat(test_pred, dim=0)
#         test_label = torch.cat(test_label, dim=0)
#
#         if args.distributed:
#             test_pred = dist_utils.gather_tensor(test_pred, args)
#             test_label = dist_utils.gather_tensor(test_label, args)
#
#         acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
#
#         if args.distributed:
#             torch.cuda.synchronize()
#
#     # Add testing results to TensorBoard
#     if val_writer is not None:
#         val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
#     # print_log('[TEST] acc = %.4f' % acc, logger=logger)
#
#     return acc