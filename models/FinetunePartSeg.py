import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from models.MaskLRF import PointTransformer

# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel
#
#     def square_distance(self, src, dst):
#         """
#         Calculate Euclid distance between each two points.
#         src^T * dst = xn * xm + yn * ym + zn * zm；
#         sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#         sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#         dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#              = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#         Input:
#             src: source points, [B, N, C]
#             dst: target points, [B, M, C]
#         Output:
#             dist: per-point square distance, [B, N, M]
#         """
#         return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)
#
#     def index_points(self, points, idx):
#         """
#         Input:
#             points: input points data, [B, N, C]
#             idx: sample index data, [B, S, [K]]
#         Return:
#             new_points:, indexed points data, [B, S, [K], C]
#         """
#         raw_size = idx.size()
#         idx = idx.reshape(raw_size[0], -1)
#         res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
#         return res.reshape(*raw_size, -1)
#
#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)
#
#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = self.square_distance(xyz1, xyz2)
#
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(self.index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
#
#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points
#
#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points
#
# # finetune model
# @MODELS.register_module()
# class PointTransformerForSegmentation(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.config = config
#
#         self.trans_dim = config.trans_dim
#         self.cls_dim = config.cls_dim
#
#         self.MAE_encoder = PointTransformer( config, **kwargs )
#
#         num_cls = 16
#         dim_global_feat = 256
#         dim_cls_label_feat = 64
#         dim_prop_feat = 1024
#         self.partseg_gf_embedder = nn.Sequential(
#             nn.Linear(self.trans_dim * self.config.depth, dim_global_feat)
#         )
#
#         self.partseg_label_conv = nn.Sequential(nn.Conv1d(num_cls, dim_cls_label_feat, kernel_size=1, bias=False),
#                                    nn.BatchNorm1d(dim_cls_label_feat),
#                                    nn.LeakyReLU(0.2))
#
#         self.partseg_propagation = PointNetFeaturePropagation(in_channel=self.trans_dim * self.config.depth + 1,
#                                                               mlp=[ self.trans_dim * 4, dim_prop_feat])
#         # self.partseg_propagation = PointNetFeaturePropagation(in_channel=self.trans_dim * self.config.depth,
#         #                                                       mlp=[ self.trans_dim * 4, dim_prop_feat])
#
#         self.partseg_conv1 = nn.Conv1d( dim_global_feat+dim_cls_label_feat+dim_prop_feat, 512, 1)
#         self.dp1 = nn.Dropout(0.5)
#         self.partseg_conv2 = nn.Conv1d(512, 256, 1)
#         self.partseg_conv3 = nn.Conv1d(256, self.cls_dim, 1)
#         self.partseg_bn1 = nn.BatchNorm1d(512)
#         self.partseg_bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
#
#         self.build_loss_func()
#
#     def build_loss_func(self):
#         self.loss_ce = nn.CrossEntropyLoss()
#
#     def get_loss_acc(self, ret, gt):
#         loss = self.loss_ce(ret, gt.long())
#         pred = ret.argmax(-1)
#         acc = (pred == gt).sum() / float(gt.size(0))
#         return loss, acc * 100
#
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         if bert_ckpt_path is not None:
#             ckpt = torch.load(bert_ckpt_path)
#             base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#
#             for k in list(base_ckpt.keys()):
#                 # if k.startswith("gf_embedder") :
#                 #     base_ckpt[ "MAE_encoder." + k ] = base_ckpt[ k ]
#                 #     del base_ckpt[k]
#                 if k.startswith('base_model'):
#                     base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#
#             incompatible = self.load_state_dict(base_ckpt, strict=False)
#
#             if incompatible.missing_keys:
#                 print_log('missing_keys', logger='Transformer')
#                 print_log(
#                     get_missing_parameters_message(incompatible.missing_keys),
#                     logger='Transformer'
#                 )
#             if incompatible.unexpected_keys:
#                 print_log('unexpected_keys', logger='Transformer')
#                 print_log(
#                     get_unexpected_parameters_message(incompatible.unexpected_keys),
#                     logger='Transformer'
#                 )
#
#             print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
#         else:
#             print_log('Training from scratch!!!', logger='Transformer')
#             self.apply(self._init_weights)
#
#     def forward( self, pts, cls_label ):
#
#         B, N, C = pts.shape
#         num_class = cls_label.shape[2]
#
#         tokens, centers, _, _ = self.MAE_encoder( pts, return_tokens=True )
#         tokens = torch.cat( tokens, dim=2 )
#
#         global_feats = tokens.mean(1)
#         global_feats = self.partseg_gf_embedder( global_feats )
#         global_feats = global_feats.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
#
#         cls_label_one_hot = cls_label.view(B, num_class, 1)
#         cls_label_feats = self.partseg_label_conv(cls_label_one_hot).repeat(1, 1, N)
#         global_feats = torch.cat( [ global_feats, cls_label_feats ], 1 )
#
#         # use distance from origin as pointwise feature
#         norm = torch.linalg.norm( pts[:,:,0:3], dim=2, keepdim=True )
#         f_level_0 = self.partseg_propagation( pts[:,:,0:3].transpose(-1, -2),
#                                               centers[:,:,0:3].transpose(-1, -2),
#                                               norm.transpose(-1, -2), tokens.transpose(-1, -2) )
#
#         # f_level_0 = self.partseg_propagation( pts[:,:,0:3].transpose(-1, -2),
#         #                                       centers[:,:,0:3].transpose(-1, -2),
#         #                                       None, tokens.transpose(-1, -2) )
#
#         x = torch.cat((f_level_0, global_feats), 1)
#         x = self.relu(self.partseg_bn1(self.partseg_conv1(x)))
#         x = self.dp1(x)
#         x = self.relu(self.partseg_bn2(self.partseg_conv2(x)))
#         x = self.partseg_conv3(x)
#         x = x.permute(0, 2, 1)
#
#         return x

# class LRF(nn.Module): # Local Reference Frames
#     def __init__(self, axis1="pca", axis2="pca"):
#         super().__init__()
#         self.axis1 = axis1
#         self.axis2 = axis2
#
#     def disambiguate_vector_directions(self, lps, vecs) :
#         # disambiguate sign of normals in the SHOT manner
#         # the codes below are borrowed from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_normals.html
#         # lps: local point sets [B,G,M,3]
#         # vecs: reference vectors [B,G,3]
#         knn = lps.shape[2]
#         proj = ( vecs[:, :, None] * lps ).sum(3) # projection of the difference on the principal direction
#         n_pos = (proj > 0).to(torch.float32).sum(2, keepdim=True) # check how many projections are positive
#         # flip the principal directions where number of positive correlations
#         flip = (n_pos < (0.5 * knn)).to(torch.float32) # 0 or 1
#         vecs = (1.0 - 2.0 * flip) * vecs # convert to -1 or 1 before multiplication with vecs
#         return vecs
#
#     def forward(self, neighbor):
#         '''
#             input:
#             neighbor: B G S 3or6 (Local point sets, whose coordinates are normalized for each local region.)
#             center: B G 3or6
#             ---------------------------
#             outputs
#             rot_neighbor: B G S 3or6 (Rotation-normalized local point sets.)
#             lrf : B G 3 3 (Local reference frames)
#         '''
#         B, G, S, C = neighbor.shape # B: batch_size, G: num_group, S: group_size
#         pos = neighbor[ :, :, :, 0:3 ]
#
#         if( C == 3 or self.axis1 == "pca" or self.axis2 == "pca" ): # in the case that PCA is necessary
#             # generate covariance matrices
#             norms = torch.linalg.norm( pos, dim=3, keepdims=True )
#             max_norms, _ = torch.max( norms, dim=2, keepdims=True )
#             w = max_norms - norms
#             w = w / ( torch.sum( w, dim=2, keepdims=True ) + 1e-6 )
#             scaled_pos = 100.0 * pos # for numerical stability
#             covs = torch.einsum( "bijk,bijl->bikl", w * scaled_pos, scaled_pos )
#
#             # There are multiple functions for eigen value decomposition
#             # Option 1
#             # _, _, eigvec = torch.linalg.svd( covs, full_matrices=False )
#             # eigvec = torch.flip( eigvec, dims=[2]).permute(0,1,3,2) # create same format as torch.linalg.eigh
#             # Option 2
#             # _, eigvec = torch.linalg.eigh( covs )
#             # Option 3
#             _, eigvec = symeig3x3( covs, eigenvectors=True )
#
#             # eigvec: [B, , 3, 3], where [:, i, :, 0] corresponds to the normal vector for the local point set i
#
#         # Compute the first axis (z_axis)
#         if( C == 3 or self.axis1 == "pca" ):
#             # z_axis is a surface normal estimated by PCA
#             z_axis = self.disambiguate_vector_directions( pos, eigvec[ :, :, :, 0 ] )
#             axis1_pca = True
#         elif( self.axis1 == "normal" ):
#             # z_axis is a true surface normal computed from polygonal 3D shape
#             z_axis = neighbor[ :, :, 0, 3:6 ] # In the "neighbor" tensor, center point always locates at the 0-th in the third axis
#             axis1_pca = False
#
#         # Compute the second axis (x_axis)
#         if( self.axis2 == "pca" ):
#             x_axis = eigvec[ :, :, :, 2 ] # eigen vectors associated with the largest eigen values
#             if( not axis1_pca ): # need to orthogonalize
#                 # each principal axis is projected onto the tangent plane of a z-axis
#                 dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
#                 x_axis = x_axis - dot * z_axis
#                 x_axis = F.normalize( x_axis, dim=2 )
#             x_axis = self.disambiguate_vector_directions( pos, x_axis )
#         elif( self.axis2 == "mean" ):
#             x_axis = torch.mean( pos, axis=2 ) # subtraction by center is not necessary since the center coordinates are always (0,0,0).
#             norm = torch.linalg.norm( x_axis, axis=2, keepdim=True )
#             x_axis = x_axis / ( norm + 1e-6 )
#             # each mean vector is projected onto the tangent plane of a z-axis
#             dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
#             x_axis = x_axis - dot * z_axis
#             x_axis = F.normalize( x_axis, dim=2 )
#
#         # Compute the third axis (y-axis), which is just a cross between z and x
#         y_axis = torch.cross( z_axis, x_axis, dim=2 )
#
#         # cat to form the set of principal directions
#         # lrfs = torch.stack( ( z_axis, y_axis, x_axis ), dim=3 )
#         lrfs = torch.stack( ( z_axis, y_axis, x_axis ), dim=-2 )
#
#         return lrfs

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def get_k_nn(xyz1, xyz2, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)

    dists, inds = dists[:, :, :k], inds[:, :, :k]
    return dists, inds

def interpolate(xyz1, xyz2, feature, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)   N1>N2
    :param feature: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = feature.shape
    dists, inds = get_k_nn(xyz1, xyz2, k)

    # inversed_dists = 1.0 / (dists + 1e-8)
    #
    # weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    #
    # weight = torch.unsqueeze(weight, -1)

    interpolated_feature = gather_points(feature, inds)  # shape=(B, N1, 3, C2)

    # return interpolated_feature, inds, weight
    return interpolated_feature, inds

# ref: PaRot: Patch-Wise Rotation-Invariant Network via Feature Disentanglement and Pose Restoration
# based on: https://github.com/dingxin-zhang/PaRot
class FP_Module_angle(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_angle, self).__init__()

        dim_posembed = 32
        self.posembed = nn.Sequential(
            nn.Conv2d( 3+1, dim_posembed, kernel_size=1, bias=False),
            nn.BatchNorm2d( dim_posembed ),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.backbone = nn.Sequential()
        bias = False if bn else True

        in_channels = in_channels + dim_posembed
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz1, xyz2, feat2, lrf2, k=3):

        B, N1, _ = xyz1.shape
        _, N2, C2 = feat2.shape

        interpolated_feature, inds = interpolate(xyz1, xyz2, feat2, k) # get features of neighboring points

        lrf2 = lrf2.reshape( B, N2, 9 )
        close_lrf = gather_points( xyz2, inds )
        lrf2 = gather_points( lrf2, inds ).view(-1, 3, 3)

        relate_position = xyz1.unsqueeze(2).repeat(1, 1, k, 1) - close_lrf

        for_dot = F.normalize(relate_position.view(-1, 3), dim=-1).unsqueeze(2)
        angle = lrf2.matmul(for_dot)
        angle = angle.view(B, N1, k, -1)

        relative_pos = torch.cat((torch.norm(relate_position, dim=-1, keepdim=True), angle), dim=3)
        pos = self.posembed(relative_pos.permute(0, 3, 2, 1))
        interpolated_feature = interpolated_feature.permute(0, 3, 2, 1)
        cat_interpolated_points = torch.cat((interpolated_feature, pos), dim=1)

        new_points = self.backbone(cat_interpolated_points)
        new_points = torch.sum(new_points, dim=2)

        return new_points

# finetune model
@MODELS.register_module()
class PointTransformerForSegmentation(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.cls_dim = config.cls_dim

        self.MAE_encoder = PointTransformer( config, **kwargs )

        num_cls = 16
        dim_global_feat = 256
        dim_token_feat = 256
        dim_cls_label_feat = 64
        dim_prop_feat = 1024
        self.partseg_gf_embedder = nn.Sequential(
            nn.Linear(self.trans_dim * self.config.depth, dim_global_feat),
            nn.BatchNorm1d(dim_global_feat),
            nn.ReLU(inplace=True)
        )
        self.partseg_token_embedder = nn.Sequential(
            nn.Linear(self.trans_dim * self.config.depth, dim_token_feat),
            nn.BatchNorm1d(dim_token_feat),
            nn.ReLU(inplace=True)
        )

        self.partseg_label_conv = nn.Sequential(nn.Conv1d(num_cls, dim_cls_label_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(dim_cls_label_feat),
                                   nn.ReLU(inplace=True))

        self.partseg_propagation = FP_Module_angle( in_channels=dim_token_feat,
                                                    mlp=[ dim_token_feat * 2, dim_prop_feat ] )

        self.partseg_conv1 = nn.Conv1d( dim_global_feat+dim_cls_label_feat+dim_prop_feat, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.partseg_conv2 = nn.Conv1d(512, 256, 1)
        self.partseg_conv3 = nn.Conv1d(256, self.cls_dim, 1)
        self.partseg_bn1 = nn.BatchNorm1d(512)
        self.partseg_bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                # if k.startswith("gf_embedder") :
                #     base_ckpt[ "MAE_encoder." + k ] = base_ckpt[ k ]
                #     del base_ckpt[k]
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
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

    def forward( self, pts, cls_label ):

        B, N, C = pts.shape
        num_class = cls_label.shape[2]

        tokens, centers, _, _, lrf = self.MAE_encoder( pts, return_tokens=True )
        tokens = torch.cat( tokens, dim=2 )

        global_feats = tokens.mean(1)
        global_feats = self.partseg_gf_embedder( global_feats )
        global_feats = global_feats.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(B, num_class, 1)
        cls_label_feats = self.partseg_label_conv(cls_label_one_hot).repeat(1, 1, N)
        global_feats = torch.cat( [ global_feats, cls_label_feats ], 1 )

        lrf = lrf.permute( 0, 1, 3, 2 ) # After permutation, the axes correspond to the rows of each 3x3 matrix

        tokens = self.partseg_token_embedder( tokens )

        f_level_0 = self.partseg_propagation( pts[:,:,0:3], centers[:,:,0:3],
                                              tokens, lrf )

        x = torch.cat((f_level_0, global_feats), 1)
        x = self.relu(self.partseg_bn1(self.partseg_conv1(x)))
        x = self.dp1(x)
        x = self.relu(self.partseg_bn2(self.partseg_conv2(x)))
        x = self.partseg_conv3(x)
        x = x.permute(0, 2, 1)

        return x
