import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from torchvision import transforms
from datasets import data_transforms

import cv2
import numpy as np

rotateSO3 = transforms.Compose( [ data_transforms.PointcloudRotateSO3() ] )

validation_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudEstimateSurfaceNormals(),
    ]
)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)

    _, _, num_train_samples = builder.dataset_builder(args, config.dataset.train)
    config.model.num_train_samples = num_train_samples

    _, test_dataloader, _ = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    data_path = args.experiment_path + "/pointsets"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, instance_ids, data) in enumerate(test_dataloader):

            points = data.cuda()
            if( config.rot_test == "so3" ):
                points = rotateSO3( points )
            points = validation_transforms( points )

            # rec_points, vis_points, centers= base_model(points, vis=True)
            orig_points, vis_points, rec_points = base_model(points, vis=True)

            taxonomy_ids = taxonomy_ids.detach().cpu().numpy()
            taxonomy_ids = taxonomy_ids[0]

            orig_points = orig_points.detach().cpu().numpy()
            np.savetxt(os.path.join( data_path, str(taxonomy_ids) + "_" + str(idx)+"_orig.xyz" ), orig_points, delimiter=' ' )

            vis_points = vis_points.detach().cpu().numpy()
            np.savetxt(os.path.join( data_path, str(taxonomy_ids) + "_" + str(idx)+"_vis.xyz" ), vis_points, delimiter=' ' )

            rec_points = rec_points.detach().cpu().numpy()
            np.savetxt(os.path.join( data_path, str(taxonomy_ids) + "_" + str(idx)+"_rec.xyz" ), rec_points, delimiter=' ' )

            if idx > 300:
                break

    return
