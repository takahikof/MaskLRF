import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from models.MaskLRF import MaskLRF

# finetune model
@MODELS.register_module()
class MaskLRFForDomainAdaptation(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.cls_dim = config.cls_dim
        self.trans_dim = config.transformer_config.trans_dim

        self.masklrf = MaskLRF( config, **kwargs )

        # classification head
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * config.transformer_config.depth, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func_rec()
        self.build_loss_func_cls()


    def build_loss_func_cls(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def build_loss_func_rec(self):
        # self.loss_ce = nn.CrossEntropyLoss()
        # reconstruction loss
        if self.config.loss == "cdl1":
            self.rec_metric = 1 # L1 distance
        elif self.config.loss =='cdl2':
            self.rec_metric = 2 # L2 distance
        else:
            raise NotImplementedError

    def get_loss_acc(self, ret, gt):
        # cross-entropy with label smoothing
        eps = 0.2
        gt = gt.to(torch.int64)
        one_hot = torch.zeros_like(ret).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (self.cls_dim - 1)
        log_prb = F.log_softmax(ret, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()

        # # cross-entropy
        # loss = self.loss_ce(ret, gt.long())

        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def get_loss_acc_mixup(self, ret, label1, label2, lambda1 ):
        # cross-entropy with mixup and label smoothing
        gt1 = F.one_hot( label1.to(torch.int64), self.cls_dim )
        gt2 = F.one_hot( label2.to(torch.int64), self.cls_dim )
        lambda1 = torch.from_numpy( np.asarray( lambda1, dtype=np.float32 ) ).clone().to( ret.device )
        lambda1 = lambda1.reshape( -1, 1 )

        gt = lambda1 * gt1 + ( 1.0 - lambda1 ) * gt2

        # eps = 0.2
        # gt = gt * (1 - eps) + (1 - gt) * eps / (self.cls_dim - 1)

        log_prb = F.log_softmax(ret, dim=1)
        loss = -(gt * log_prb).sum(dim=1).mean()

        # # cross-entropy
        # loss = self.loss_ce(ret, gt.long())

        pred = ret.argmax(-1)
        acc = (pred == label1).sum() / float(label1.size(0))
        return loss, acc * 100

    def compute_loss_rec(self, rebuild_points, gt_points):

        # reconstruction loss
        if( self.config.output_feature_type == "point_only" ):
            rec_loss = chamfer_distance( rebuild_points, gt_points, norm=self.rec_metric )
            rec_loss = rec_loss[0]
        elif( self.config.output_feature_type == "point_normal" ):
            rec_loss = chamfer_distance( x=rebuild_points[:,:,0:3],
                                         y=gt_points[:,:,0:3],
                                         x_normals=rebuild_points[:,:,3:6],
                                         y_normals=gt_points[:,:,3:6],
                                         norm=self.rec_metric, abs_cosine=True )
            rec_loss = rec_loss[0] + 0.1 * rec_loss[1]
        elif( self.config.output_feature_type == "occupancy_grid" ):
            rec_loss = F.binary_cross_entropy( torch.sigmoid( rebuild_points ), gt_points )
        elif( self.config.output_feature_type == "normal_curvature" or
              self.config.output_feature_type == "fpfh" or
              self.config.output_feature_type == "pod" ):
            if( self.rec_metric == 1 ):
                rec_loss = torch.abs( rebuild_points - gt_points ).sum(1).mean()
            elif( self.rec_metric == 2 ): # MSE error
                rec_loss = torch.square( rebuild_points - gt_points ).sum(1).mean()

        return rec_loss

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                base_ckpt[ "masklrf." + k ] = base_ckpt[k]
                del base_ckpt[k]

            for k in list(base_ckpt.keys()):
                if k.startswith('masklrf.increase_dim') : # 再構成ターゲットが違うので読み込まない
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def forward( self, pts, cls=False ):
        if( cls ):
            global_feats = self.masklrf( pts, extract_global_features=True )
            ret = self.cls_head_finetune( global_feats ) # classification logits
        else:
            ret = self.masklrf( pts, extract_global_features=False ) # reconstruction loss
        return ret
