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

train_transforms = transforms.Compose(
    [
         data_transforms.PointcloudScaleAnisotropic(scale_low=0.8, scale_high=1.2),
    ]
)

# estimateNormals = transforms.Compose( [ data_transforms.PointcloudEstimateSurfaceNormals() ] )

# test_transforms = transforms.Compose(
#     [
#         data_transforms.PointcloudScaleAnisotropic(scale_low=0.8, scale_high=1.2),
#     ]
# )

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

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

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def run_net(args, config, train_writer=None, val_writer=None):

    logger = get_logger(args.log_name)
    # build dataset
    train_sampler, train_dataloader, _ = builder.dataset_builder(args, config.dataset.train) # trainval
    _, test_dataloader, _ = builder.dataset_builder(args, config.dataset.val) # test

    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics_iiou = Acc_Metric(0.)
    best_metrics_ciou = Acc_Metric(0.)
    best_epoch_iiou = 0
    best_epoch_ciou = 0
    metrics_iiou = Acc_Metric(0.)
    metrics_ciou = Acc_Metric(0.)
    # best_metrics_vote = Acc_Metric(0.)

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
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (points, object_label, point_label) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            # for i in range( data[0].shape[0] ):
            #     out_filepath = "../data_downloaded/xyz_samples/ModelNetFewshot/" + str( i ) + ".xyz"
            #     np.savetxt( out_filepath, data[0][i] )
            # quit()

            points = points.cuda()
            object_label = object_label.cuda()
            point_label = point_label.cuda()

            if( config.rot_train == "so3" ):
                points = rotateSO3( points )

            # # we do not use surface normals
            # points = points[ :, :, 0:3 ]
            # points = estimateNormals( points )

            points = train_transforms( points )

            # for i in range( points.shape[0] ):
            #     x = points[i].to('cpu').detach().numpy().copy()
            #     out_filepath = "out/" + str( i ) + ".xyz"
            #     np.savetxt( out_filepath, x )
            # quit()

            # print( "----------------")
            # start_time = time.time()
            ret = base_model( points, to_categorical(object_label, config.num_classes) )
            # print( "infer: " + str( time.time() - start_time ) )

            B, P, _ = ret.shape
            ret = ret.reshape( B*P, -1 )
            point_label = point_label.reshape( B*P )

            loss, acc = base_model.module.get_loss_acc(ret, point_label)
            _loss = loss

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
                losses.update([loss.item(), acc.item()])


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
            metrics_ciou, metrics_iiou = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better_ciou = metrics_ciou.better_than(best_metrics_ciou)
            better_iiou = metrics_iiou.better_than(best_metrics_iiou)

            # Save ckeckpoints
            if better_ciou:
                best_metrics_ciou = metrics_ciou
                best_epoch_ciou = epoch
                # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if better_iiou:
                best_metrics_iiou = metrics_iiou
                best_epoch_iiou = epoch
                print_log("--------------------------------------------------------------------------------------------", logger=logger)

            print_log('Current best Class avg. IoU = %.4f at %d th epoch' % ( best_metrics_ciou.acc, best_epoch_ciou ), logger = logger)
            print_log('Current best Instance avg. IoU = %.4f at %d th epoch' % ( best_metrics_iiou.acc, best_epoch_iiou ), logger = logger)

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

    num_part = config.num_part

    test_metrics = {}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    with torch.no_grad():
        for idx, (points, object_label, point_label) in enumerate(test_dataloader):
            cur_batch_size, NUM_POINT, _ = points.size()
            points = points.cuda()
            object_label = object_label.cuda()
            point_label = point_label.cuda()

            if( config.rot_test == "so3" ):
                points = rotateSO3( points )

            # # we do not use surface normals
            # points = points[ :, :, 0:3 ]
            # points = estimateNormals( points )

            seg_pred = base_model( points, to_categorical(object_label, config.num_classes) )

            target = point_label
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            print_log('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]), logger=logger)
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    print_log('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']), logger=logger)
    return Acc_Metric( test_metrics['class_avg_iou'] ), Acc_Metric( test_metrics['inctance_avg_iou'] )
