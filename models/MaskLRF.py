import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from pytorch3d.common.workaround import symeig3x3
from pytorch3d.ops import sample_farthest_points, knn_gather
from models.chamfer_distance import chamfer_distance
from models.pod_extractor import HandcraftedPodExtractor
from models.fpfh_extractor import HandcraftedFPFHExtractor
from models.normal_curvature_extractor import NormalCurvatureExtractor
from models.occupancy_grid_extractor import OccupancyGridExtractor

class LocalPointSetEncoder(nn.Module):   # Embedding module
    def __init__(self, config, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel

        self.first_conv = nn.Sequential(
            nn.Conv1d( 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 6
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape

        point_groups = point_groups[ :, :, :, 0:3 ] # discard normals
        point_groups = point_groups.reshape(bs * g, n, -1)
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024

        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, max_group_size=128):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.max_group_size = max_group_size

    def nn_search( self, Q_pts, T_pts, knn ) :
        # Q_pts : an input batch of query 3D points [B, P1, 3]
        # T_pts : an input batch of target 3D point sets [B, P2, 3]

        # The knn_points function in Pytorch3D is slow when knn is very large
        # _, nn_idx, _ = knn_points( Q_pts, T_pts, K=knn ) # [B, P1, K]

        distmats = torch.cdist( Q_pts, T_pts )
        dists, nn_idx = torch.topk( distmats, knn, dim=2, largest=False, sorted=True )
        return nn_idx

    def forward(self, pc):
        '''
            input: B N 3or6
            ---------------------------
            outputs
            neighborhood : B G S 3or6
            center : B G 3or6
        '''
        batch_size, num_points, C = pc.shape
        pos = pc[ :, :, 0:3 ]
        # ori = pc[ :, :, 3:6 ]

        # Choose center points by using FPS
        _, rep_idx = sample_farthest_points( pos, K=self.num_group, random_start_point=True )
        rep_idx = torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,C)
        center = torch.gather( pc, dim=1, index=rep_idx )

        # Find neighboring points of each center point
        nn_idx = self.nn_search( center[ :, :, 0:3 ], pos, self.group_size )
        neighborhood = knn_gather( pc, nn_idx )

        # normalize position
        neighborhood[ :, :, :, 0:3 ] = neighborhood[ :, :, :, 0:3 ] - center.unsqueeze(2)[ :, :, :, 0:3 ]

        return neighborhood, center, nn_idx

    def get_actual_group_size( self ):
        if( self.group_size > self.max_group_size ) :
            return self.max_group_size
        else :
            return self.group_size

class LRF(nn.Module): # Local Reference Frames
    def __init__(self, axis1="pca", axis2="pca"):
        super().__init__()
        self.axis1 = axis1
        self.axis2 = axis2

    def disambiguate_vector_directions(self, lps, vecs) :
        # disambiguate sign of normals in the SHOT manner
        # the codes below are borrowed from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_normals.html
        # lps: local point sets [B,G,M,3]
        # vecs: reference vectors [B,G,3]
        knn = lps.shape[2]
        proj = ( vecs[:, :, None] * lps ).sum(3) # projection of the difference on the principal direction
        n_pos = (proj > 0).to(torch.float32).sum(2, keepdim=True) # check how many projections are positive
        # flip the principal directions where number of positive correlations
        flip = (n_pos < (0.5 * knn)).to(torch.float32) # 0 or 1
        vecs = (1.0 - 2.0 * flip) * vecs # convert to -1 or 1 before multiplication with vecs
        return vecs

    def forward(self, neighbor, center):
        '''
            input:
            neighbor: B G S 3or6 (Local point sets, whose coordinates are normalized for each local region.)
            center: B G 3or6
            ---------------------------
            outputs
            rot_neighbor: B G S 3or6 (Rotation-normalized local point sets.)
            lrf : B G 3 3 (Local reference frames)
        '''
        B, G, S, C = neighbor.shape # B: batch_size, G: num_group, S: group_size
        pos = neighbor[ :, :, :, 0:3 ]

        if( C == 3 or self.axis1 == "pca" or self.axis2 == "pca" ): # in the case that PCA is necessary
            # generate covariance matrices
            norms = torch.linalg.norm( pos, dim=3, keepdims=True )
            max_norms, _ = torch.max( norms, dim=2, keepdims=True )
            w = max_norms - norms
            w = w / ( torch.sum( w, dim=2, keepdims=True ) + 1e-6 )
            scaled_pos = 100.0 * pos # for numerical stability
            covs = torch.einsum( "bijk,bijl->bikl", w * scaled_pos, scaled_pos )

            # There are multiple functions for eigen value decomposition
            # Option 1
            # _, _, eigvec = torch.linalg.svd( covs, full_matrices=False )
            # eigvec = torch.flip( eigvec, dims=[2]).permute(0,1,3,2) # create same format as torch.linalg.eigh
            # Option 2
            # _, eigvec = torch.linalg.eigh( covs )
            # Option 3
            _, eigvec = symeig3x3( covs, eigenvectors=True )

            # eigvec: [B, , 3, 3], where [:, i, :, 0] corresponds to the normal vector for the local point set i

        # Compute the first axis (z_axis)
        if( C == 3 or self.axis1 == "pca" ):
            # z_axis is a surface normal estimated by PCA
            z_axis = self.disambiguate_vector_directions( pos, eigvec[ :, :, :, 0 ] )
            axis1_pca = True
        elif( self.axis1 == "normal" ):
            # z_axis is a true surface normal computed from polygonal 3D shape
            z_axis = neighbor[ :, :, 0, 3:6 ] # In the "neighbor" tensor, center point always locates at the 0-th in the third axis
            axis1_pca = False

        # Compute the second axis (x_axis)
        if( self.axis2 == "pca" ):
            x_axis = eigvec[ :, :, :, 2 ] # eigen vectors associated with the largest eigen values
            if( not axis1_pca ): # need to orthogonalize
                # each principal axis is projected onto the tangent plane of a z-axis
                dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
                x_axis = x_axis - dot * z_axis
                x_axis = F.normalize( x_axis, dim=2 )
            x_axis = self.disambiguate_vector_directions( pos, x_axis )
        elif( self.axis2 == "mean" ):
            x_axis = torch.mean( pos, axis=2 ) # subtraction by center is not necessary since the center coordinates are always (0,0,0).
            norm = torch.linalg.norm( x_axis, axis=2, keepdim=True )
            x_axis = x_axis / ( norm + 1e-6 )
            # each mean vector is projected onto the tangent plane of a z-axis
            dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
            x_axis = x_axis - dot * z_axis
            x_axis = F.normalize( x_axis, dim=2 )

        # Compute the third axis (y-axis), which is just a cross between z and x
        y_axis = torch.cross( z_axis, x_axis, dim=2 )

        # cat to form the set of principal directions
        lrfs = torch.stack( ( z_axis, y_axis, x_axis ), dim=3 )

        # normalize orientation of local point sets
        pos = torch.reshape( pos, [ B * G, S, 3 ] )
        pos = torch.bmm( pos, torch.reshape( lrfs, [ B * G, 3, 3 ] ) )
        pos = torch.reshape( pos, [ B, G, S, 3 ] )

        if( C == 3 ):
            rot_neighbor = pos
        elif( C == 6 ):
            ori = neighbor[ :, :, :, 3:6 ]
            ori = torch.reshape( ori, [ B * G, S, 3 ] )
            ori = torch.bmm( ori, torch.reshape( lrfs, [ B * G, 3, 3 ] ) )
            ori = torch.reshape( ori, [ B, G, S, 3 ] )
            rot_neighbor = torch.cat( [ pos, ori ], dim=3 )

        return rot_neighbor, lrfs

class RelativePosOriEncoder(nn.Module): # Relative position/orientation encoding
    def __init__(self):
        super().__init__()

    def compute_relative_position_encoding( self, center, lrf ):
        B, G, _ = center.shape
        center_pos = center[ :, :, 0:3 ]
        # relative positions in the ambient 3D space
        rel_pos = center_pos[ :, None, :, : ] - center_pos[ :, :, None, : ]
        # relative positions in the local reference frame
        rel_pos = torch.reshape( rel_pos, [ B * G, G, 3 ] )
        rel_pos = torch.bmm( rel_pos, torch.reshape( lrf, [ B * G, 3, 3 ] ) )
        rel_pos = torch.reshape( rel_pos, [ B, G, G, 3 ] )
        return rel_pos

    def compute_relative_orientation_encoding( self, lrf ):
        # Assumes lrf[:, i, :, 0] corresponds to the 0-th axis (the normal vector) of the local reference frame i
        B, G, _, _ = lrf.shape

        # For each 3D point set, matrix multiplications are computed for all pairs of LRFs.
        lrf_from = torch.transpose( lrf, 2, 3 ) # 軸同士の内積が計算されるよう，行列積における第1引数の行列は転置する．
        lrf_from = lrf_from.unsqueeze(2)
        lrf_from = torch.tile( lrf_from, (1,1,G,1,1) )
        lrf_from = torch.reshape( lrf_from, [ B*G*G, 3, 3 ] )

        lrf_to = lrf.unsqueeze(1)
        lrf_to = torch.tile( lrf_to, (1,G,1,1,1) )
        lrf_to = torch.reshape( lrf_to, [ B*G*G, 3, 3 ] )

        rel_ori = torch.bmm( lrf_from, lrf_to )
        rel_ori = torch.reshape( rel_ori, [ B, G, G, 3, 3 ] )
        return rel_ori

    def forward( self, center, lrf ):
        '''
            input:
            center: B G 6
            lrf : B G 3 3 (Local reference frames)
            ---------------------------
            outputs
            rel_pos: B G G 3 (Relative position encoding.)
            rel_ori: B G G 9 (Relative orientation encoding.)
        '''
        # 1. Compute relative position encoding
        rel_pos = self.compute_relative_position_encoding( center, lrf )
        # 2. Compute relative orientation encoding
        rel_ori = self.compute_relative_orientation_encoding( lrf )
        return rel_pos, rel_ori

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, config, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm, depth=0, depth_rate=0. ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_relQ = norm_layer(dim)
        self.norm_relK = norm_layer(dim)
        self.norm_relV = norm_layer(dim)

        self.attn_subsample_rate = config.attn_subsample_rate
        self.attn_target = config.attn_target
        self.attn_mode = config.attn_mode
        # self.attn_mode = "no_rel" # No relative encoding
        # self.attn_mode = "rel_bias" # relative encoding with the "bias" mode
        # self.attn_mode = "rel_contextual" # relative encoding with the "contextual" mode
        # self.attn_mode = "vector" # Vector self-attention
        # "rel_bias" and "rel_contextual follows the idea of "Rethinking and Improving Relative Position Encoding for Vision Transformer"

        if( self.attn_mode == "rel_bias" ):
            self.to_scalar = nn.Linear( dim, num_heads )
        elif( self.attn_mode == "vector" ):
            self.attention_mlp = nn.Sequential( nn.Linear( dim, dim ),
                                                nn.ReLU(),
                                                nn.Linear( dim, dim ) )

        self.depth = depth
        self.depth_rate = depth_rate

    def forward( self, x, rel, rel_embedder ):
        B, N, C = x.shape

        Nsub = int( np.round( N * self.attn_subsample_rate ) )
        centers = rel[ :, 0, :, 0:3 ] # 近傍を調べるため，0番目(何番目でもよいが)のLRF上の代表点群を抜き出す
        distmats = torch.cdist( centers, centers )

        if( self.attn_target == "knn_only" or N < 10 ):
            # どの層においても k近傍の代表点群をアテンション対象として用いる
            _, nn_idx = torch.topk( distmats, Nsub, dim=2, largest=False, sorted=True )
        elif( self.attn_target == "dilate_only" ):
            _, nn_idx = sample_farthest_points( centers, K=Nsub, random_start_point=True )
            nn_idx = nn_idx.unsqueeze(1).repeat(1,N,1)
        elif( self.attn_target == "knn_dilate" ):
            # Inspired by the paper "Dilated Neighborhood Attention Transformer"
            if( self.depth % 2 == 0 ): # Local self-attention
                _, nn_idx = torch.topk( distmats, Nsub, dim=2, largest=False, sorted=True )
            else: # dilated self-attention
                # FPS is used since attention targets should distribute uniformly.
                _, nn_idx = sample_farthest_points( centers, K=Nsub, random_start_point=True )
                nn_idx = nn_idx.unsqueeze(1).repeat(1,N,1)
        elif( self.attn_target == "random" ):
            nn_idx = []
            for i in range( N ): # for each token
                idx = np.arange( N )
                np.random.shuffle( idx )
                nn_idx.append( np.reshape( idx[0:Nsub], (1,Nsub) ) )
            nn_idx = np.vstack( nn_idx )
            nn_idx = np.reshape( nn_idx, ( 1, N, Nsub ) )
            nn_idx = np.tile( nn_idx, ( B, 1, 1 ) )
            nn_idx = torch.from_numpy( nn_idx.astype(np.int64) ).clone()
            nn_idx = nn_idx.to( x.device )

        qkv = self.qkv(x)
        qkv = torch.chunk( qkv, 3, dim=2 )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = knn_gather( k, nn_idx )
        v = knn_gather( v, nn_idx )

        rel = torch.gather( rel, dim=2, index=nn_idx.unsqueeze(3).repeat(1,1,1,3+9) )
        rel = rel.reshape( B*N*Nsub, 3+9 )
        relQ = rel_embedder( rel )
        relV = relQ

        if( self.attn_mode == "no_rel" ): # standard, scaled dot-product self-attention
            q = q.reshape( B, N, 1, self.num_heads, C // self.num_heads )
            q = q.permute(0,3,1,2,4)
            k = k.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            k = k.permute(0,3,1,2,4)
            attn = torch.mul( q, k ).sum(-1)
            attn *= self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            v = v.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            v = v.permute(0,3,1,2,4)
            v = torch.mul( attn.unsqueeze(-1), v ) # apply attention weights
            x = torch.sum( v, dim=3 )
            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

        elif( self.attn_mode == "rel_bias" ): # with relative encoding (the bias mode)
            # biasモードの相対埋込(rel_emb)は，入力特徴xに依存しないスカラー値
            rel_emb = self.to_scalar( relQ.reshape( B*N*Nsub, C ) ).reshape( B, N, Nsub, self.num_heads )
            rel_emb = rel_emb.permute(0,3,1,2)

            q = q.reshape( B, N, 1, self.num_heads, C // self.num_heads )
            q = q.permute(0,3,1,2,4)
            k = k.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            k = k.permute(0,3,1,2,4)

            attn = torch.mul( q, k ).sum(-1)
            attn += rel_emb
            attn *= self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            v = v.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            v = v.permute(0,3,1,2,4)
            v = torch.mul( attn.unsqueeze(-1), v ) # apply attention weights
            x = torch.sum( v, dim=3 )
            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

        elif( self.attn_mode == "rel_contextual" ): # with relative encoding (the contextual mode)
            # contextualモードの相対埋込(rel_emb)は，入力特徴xに依存するスカラー値

            # relative embedding for Query
            relQ = relQ.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            relQ = relQ.permute(0,3,1,2,4)
            q = q.reshape( B, N, 1, self.num_heads, C // self.num_heads )
            q = q.permute(0,3,1,2,4)
            relQ = torch.mul( q, relQ )
            relQ = torch.sum( relQ, dim=-1 )

            k = k.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            k = k.permute(0,3,1,2,4)

            attn = torch.mul( q, k ).sum(-1)
            attn += relQ
            attn *= self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # relative embedding for Value
            relV = relV.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            relV = relV.permute(0,3,1,2,4)
            v = v.reshape( B, N, Nsub, self.num_heads, C // self.num_heads )
            v = v.permute(0,3,1,2,4)
            v += relV
            v = torch.mul( attn.unsqueeze(-1), v ) # apply attention weights
            x = torch.sum( v, dim=3 )
            x = x.transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

        elif( self.attn_mode == "vector" ):
            q = q.reshape( B*N, 1, -1 )
            k = k.reshape( B*N, Nsub, -1 )
            v = v.reshape( B*N, Nsub, -1 )
            relQ = relQ.reshape( B*N, Nsub, -1 )
            relV = relV.reshape( B*N, Nsub, -1 )

            # code below is based on: https://github.com/lucidrains/point-transformer-pytorch/blob/main/point_transformer_pytorch/point_transformer_pytorch.py
            qk = q[:, :, None, :] - k[:, None, :, :] # [B, 1, T, C]
            qk = qk.squeeze()
            qk += relQ
            sim = self.attention_mlp( qk )
            A = sim.softmax( dim=1 ) # attention map
            v += relV
            x = torch.sum( A * v, dim=1 )
            x = x.reshape( B, N, -1 )

        return x


class Block(nn.Module):
    def __init__(self, config, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, depth_rate=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention( config,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            depth=depth, depth_rate=depth_rate )

    def forward( self, x, rel, rel_embedder ):
        x = x + self.drop_path( self.attn( self.norm1(x), rel, rel_embedder ) )
        x = x + self.drop_path( self.mlp( self.norm2(x) ) )
        return x


class Transformer(nn.Module):
    def __init__(self, config, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                config=config,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                depth = i, depth_rate = (i+1.0)/depth
                )
            for i in range(depth)])

    def forward( self, x, rel, rel_embedder ):
        features = []
        for _, block in enumerate(self.blocks):
            x = block( x, rel, rel_embedder )
            features.append( x )
        return features


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')

        # local point set embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.lps_encoder = LocalPointSetEncoder(config=self.config, encoder_channel=self.encoder_dims)

        # relative position/orientation embedding
        self.rel_embedder = nn.Sequential(
            nn.Linear(3+9, 128), nn.GELU(), nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Transformer(
            config = self.config,
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _create_mask_blockwise(self, rel_pos, noaug=False):
        '''
            rel_pos : B G G 3 (relative position encoding)
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros( rel_pos.shape[:2] ).bool()

        # mask a continuous part
        center = rel_pos[ :, 0, :, : ]
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(rel_pos.device)  # B G

        return bool_masked_pos

    def _create_mask_random(self, rel_pos, noaug=False):
        '''
            rel_pos : B G G 3 (relative position encoding)
            --------------
            mask : B G (bool)
        '''
        B, G, _, _ = rel_pos.shape

        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros( rel_pos.shape[:2] ).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(rel_pos.device) # B G

    def forward(self, neighborhood, rel_pos, rel_ori, noaug=False):
        # Generate mask
        if self.mask_type == 'rand':
            bool_mask = self._create_mask_random( rel_pos, noaug=noaug ) # B G
        else:
            bool_mask = self._create_mask_blockwise( rel_pos, noaug=noaug )

        # Compute token features
        group_input_tokens = self.lps_encoder( neighborhood )  #  B G C
        B, G, C = group_input_tokens.size()

        # Collect visible tokens
        vis_token = group_input_tokens[ ~bool_mask ].reshape( B, -1, C )
        Gsub = vis_token.shape[1]

        # Collect visible relative position/orientations
        rel_ori = rel_ori.reshape( B, G, G, 9 )
        rel = torch.cat( [ rel_pos, rel_ori ], dim=3 )
        vis_rel = rel[ ~bool_mask ].reshape( B, Gsub, G, 3+9 )
        vis_rel = vis_rel.permute(0,2,1,3)[ ~bool_mask ].reshape( B, Gsub, Gsub, 3+9 ).permute(0,2,1,3)

        # transformer
        vis_token = self.blocks( vis_token, vis_rel, self.rel_embedder )
        # vis_token = self.norm( vis_token )

        return vis_token, bool_mask


@MODELS.register_module()
class MaskLRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskLRF] ', logger ='MaskLRF')
        self.config = config
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.trans_dim = config.transformer_config.trans_dim
        self.drop_path_rate = config.transformer_config.drop_path_rate

        print_log(f'[MaskLRF] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='MaskLRF')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.lrf_estimator = LRF(axis1=self.config.lrf_axis1, axis2=self.config.lrf_axis2)
        self.relposori_encoder = RelativePosOriEncoder()

        # Encoder DNN
        self.MAE_encoder = MaskTransformer(config)

        # Decoder DNN
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # relative position/orientation embedding
        self.decoder_rel_embedder = nn.Sequential(
            nn.Linear(3+9, 128), nn.GELU(), nn.Linear(128, self.trans_dim),
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = Transformer(
            config = self.config,
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.decoder_norm = nn.LayerNorm( self.trans_dim )

        # reconstruction head
        if( config.output_feature_type == "point_only" ):
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, 3*self.group_divider.get_actual_group_size(), 1)
            )
        elif( config.output_feature_type == "point_normal" ):
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, 6*self.group_divider.get_actual_group_size(), 1)
            )
        elif( config.output_feature_type == "normal_curvature" ):
            self.desc_extractor = NormalCurvatureExtractor()
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, self.desc_extractor.get_outdim(), 1)
            )
        elif( config.output_feature_type == "occupancy_grid" ):
            self.desc_extractor = OccupancyGridExtractor( num_bins=config.pod_gridsize )
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, self.desc_extractor.get_outdim(), 1)
            )
        elif( config.output_feature_type == "fpfh" ):
            self.desc_extractor = HandcraftedFPFHExtractor( num_bins=5 )
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, self.desc_extractor.get_outdim(), 1)
            )
        elif( config.output_feature_type == "pod" ):
            self.desc_extractor = HandcraftedPodExtractor( num_bins=config.pod_gridsize )
            self.increase_dim = nn.Sequential(
                nn.Conv1d(self.trans_dim, self.desc_extractor.get_outdim(), 1)
            )

        # global feature embedding head
        self.gf_embedder = nn.Sequential(
            nn.Linear(self.trans_dim *  config.transformer_config.depth, 256)
        )

        trunc_normal_(self.mask_token, std=.02)

        # reconstruction loss
        if config.loss == "cdl1":
            self.rec_metric = 1 # L1 distance
        elif config.loss =='cdl2':
            self.rec_metric = 2 # L2 distance
        else:
            raise NotImplementedError

    def compute_loss(self, rebuild_points, gt_points):

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

    def recover_input_shapes_abs( self, gt, vis_points, rebuild_points, lrf, center, bool_mask ):
        # This function assumes batch size is 1.
        if( gt.shape[0] != 1 ):
            print( "batch size must be 1." )
            quit()

        gt = gt[0,:,0:3] # [1024,3]

        vis_points = vis_points[0,:,:]
        rebuild_points = rebuild_points[0,:,:]

        vis_lrf = lrf[ ~bool_mask ]
        masked_lrf = lrf[ bool_mask ]

        vis_center = center[ ~bool_mask ][:,0:3]
        masked_center = center[ bool_mask ][:,0:3]

        # print( gt.shape )
        # print( vis_points.shape )
        # print( rebuild_points.shape )
        # print( vis_lrf.shape )
        # print( masked_lrf.shape )
        # print( vis_center.shape )
        # print( masked_center.shape )

        # recover pose of visible tokens
        vis_points = torch.bmm( vis_points, vis_lrf.permute(0,2,1) )
        vis_points = vis_points + vis_center.unsqueeze(1)
        vis_points = vis_points.reshape(-1,3)

        # recover pose of reconstructed tokens
        rebuild_points = torch.bmm( rebuild_points, masked_lrf.permute(0,2,1) )
        rebuild_points = rebuild_points + masked_center.unsqueeze(1)
        rebuild_points = rebuild_points.reshape(-1,3)

        return gt, vis_points, rebuild_points

    def recover_input_shapes( self, gt, neighborhood_rot, rel_pos, rel_ori ):
        # Demonstrates that an original 3D shape can be recovered from:
        # * rotation-normalized local point sets (neighborhood_rot)
        # * relative position encoding (rel_pos)
        # * relative orientation encoding (rel_ori)
        B, G, S, C = neighborhood_rot.shape
        recovered_shapes = []
        for i in range( B ):

            # Save the original whole 3D shape
            out_filepath = "./out/" + str( i ) + "_original.xyz"
            np.savetxt( out_filepath, gt[ i ].to('cpu').detach().numpy().copy() )

            # Rotation-normalized local point sets are aranged in a row in the 3D space
            # and are saved in a single file.
            pc = neighborhood_rot[ i ].clone()
            for j in range( G ):
                offset = j * 2.0
                pc[ j, :, 0 ] += offset # offset the x coordinate
            pc = torch.reshape( pc, [ G*S, -1 ] )
            out_filepath = "./out/" + str( i ) + "_neighborhood_rot.xyz"
            np.savetxt( out_filepath, pc.to('cpu').detach().numpy().copy() )

            # Recover the whole 3D shape in the LRF of 0-th local point set
            pc = neighborhood_rot[ i ].clone()
            pos = pc[ :, :, 0:3 ]
            rp = rel_pos[ i, 0, :, : ]
            ro = rel_ori[ i, 0, :, :, : ]
            pos = torch.bmm( pos, ro.permute(0,2,1) ) # recover orientation of local regions
            pos += rp.unsqueeze(1) # recover position of local regions
            pc[ :, :, 0:3 ] = pos
            if( C == 6 ): # when each 3D point has a normal vector
                ori = pc[ :, :, 3:6 ]
                ori = torch.bmm( ori, ro.permute(0,2,1) )
                pc[ :, :, 3:6 ] = ori
            pc = torch.reshape( pc, [ G*S, -1 ] )
            out_filepath = "./out/" + str( i ) + "_recovered.xyz"
            np.savetxt( out_filepath, pc.to('cpu').detach().numpy().copy() )
            recovered_shapes.append( pc.unsqueeze(0) )

        recovered_shapes = torch.cat( recovered_shapes, dim=0 )
        return recovered_shapes

    def forward(self, pts,
                      pseudo_labels = None,
                      vis = False,
                      extract_global_features = False, **kwargs):

        # Tokenize
        neighborhood, center, _ = self.group_divider( pts )
        neighborhood_rot, lrf = self.lrf_estimator( neighborhood, center )
        rel_pos, rel_ori = self.relposori_encoder( center, lrf )
        # self.recover_input_shapes( pts, neighborhood_rot, rel_pos, rel_ori )
        # quit()

        if( extract_global_features ) :
            # Encode tokens (with no masking)
            vis_token, bool_mask = self.MAE_encoder( neighborhood_rot, rel_pos, rel_ori, noaug=True )
            # Get global features
            global_feats = torch.cat( vis_token, dim=2 ).mean(1) # tokens from all Transformer blocks are used
            return global_feats

        # Encode tokens (with masking)
        vis_token, bool_mask = self.MAE_encoder( neighborhood_rot, rel_pos, rel_ori, noaug=False )

        vis_token = vis_token[-1] # Tokens produced by the final transformer block are used as inputs to the decoder
        B, G, _, _ = neighborhood_rot.shape
        _, Gsub, C = vis_token.shape # Gsub is the number of visible tokens

        # Decode
        # Collect visible relative position/orientations
        rel_ori = rel_ori.reshape( B, G, G, 9 )
        rel = torch.cat( [ rel_pos, rel_ori ], dim=3 )
        # The following process is a little complicated.
        # In each relative encoding matrix, I want to place visible tokens in the first Gsub rows/columns,
        # and to place masked tokens in the subsequent (G-Gsub) rows/columns.
        _, indices = torch.sort( bool_mask.to(torch.int32), dim=-1, descending=False, stable=True )
        indices = indices.unsqueeze(2).unsqueeze(3).tile(1,1,G,3+9)
        rel = torch.gather( rel, dim=1, index=indices )
        full_rel = torch.gather( rel.permute(0,2,1,3), dim=1, index=indices ).permute(0,2,1,3)

        # Create the tensor containing both visible and masked tokens
        mask_token = self.mask_token.expand( B, G-Gsub, -1 )
        full_token = torch.cat( [ vis_token, mask_token ], dim=1 )

        # Decode
        full_token = self.MAE_decoder( full_token, full_rel, self.decoder_rel_embedder )
        full_token = full_token[-1] # Tokens produced by the final transformer block are used as inputs to the prediction
        mask_token = self.decoder_norm( full_token[ :, -(G-Gsub):] )  # get the masked tokens
        rebuild_points = self.increase_dim( mask_token.transpose(1, 2)).transpose(1, 2)

        # Create supervision signals and reshape reconstructions
        gt_points = neighborhood_rot[ bool_mask ]
        if( self.config.output_feature_type == "point_only" ):
            gt_points = gt_points[ :, :, 0:3 ]
            rebuild_points = rebuild_points.reshape( B*(G-Gsub), -1, 3 )
        elif( self.config.output_feature_type == "point_normal" ):
            rebuild_points = rebuild_points.reshape( B*(G-Gsub), -1, 6 )
        elif( self.config.output_feature_type == "normal_curvature" or
              self.config.output_feature_type == "occupancy_grid" or
              self.config.output_feature_type == "fpfh" or
              self.config.output_feature_type == "pod" ):
            gt_points = self.desc_extractor( gt_points )
            rebuild_points = rebuild_points.reshape( B*(G-Gsub), self.desc_extractor.get_outdim() )

        rec_loss = self.compute_loss( rebuild_points, gt_points )

        if vis: # visualization
            rebuild_points = rebuild_points.reshape( B, (G-Gsub), -1, 3 )

            vis_points = neighborhood_rot[~bool_mask]
            vis_points = vis_points[ :, :, 0:3 ]
            vis_points = vis_points.reshape( B, Gsub, -1, 3 )

            orig_points, vis_points, rec_points = self.recover_input_shapes_abs( pts, vis_points, rebuild_points, lrf, center, bool_mask )
            return orig_points, vis_points, rec_points
        else:
            return rec_loss

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.lrf_estimator = LRF(axis1=self.config.lrf_axis1, axis2=self.config.lrf_axis2)
        self.relposori_encoder = RelativePosOriEncoder()
        self.lps_encoder = LocalPointSetEncoder(config=self.config, encoder_channel=self.encoder_dims)

        # relative position/orientation embedding
        self.rel_embedder = nn.Sequential(
            nn.Linear(3+9, 128), nn.GELU(), nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Transformer(
            config = self.config,
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        # self.norm = nn.LayerNorm(self.trans_dim)

        # classification head
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * self.depth, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

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

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def recover_input_shapes( self, gt, neighborhood_rot, rel_pos, rel_ori ):
        # Demonstrates that an original 3D shape can be recovered from:
        # * rotation-normalized local point sets (neighborhood_rot)
        # * relative position encoding (rel_pos)
        # * relative orientation encoding (rel_ori)
        B, G, S, C = neighborhood_rot.shape
        recovered_shapes = []
        for i in range( B ):

            # Save the original whole 3D shape
            out_filepath = "./out/" + str( i ) + "_original.xyz"
            np.savetxt( out_filepath, gt[ i ].to('cpu').detach().numpy().copy() )

            # Rotation-normalized local point sets are aranged in a row in the 3D space
            # and are saved in a single file.
            pc = neighborhood_rot[ i ].clone()
            for j in range( G ):
                offset = j * 2.0
                pc[ j, :, 0 ] += offset # offset the x coordinate
            pc = torch.reshape( pc, [ G*S, -1 ] )
            out_filepath = "./out/" + str( i ) + "_neighborhood_rot.xyz"
            np.savetxt( out_filepath, pc.to('cpu').detach().numpy().copy() )

            # Recover the whole 3D shape in the LRF of 0-th local point set
            pc = neighborhood_rot[ i ].clone()
            pos = pc[ :, :, 0:3 ]
            rp = rel_pos[ i, 0, :, : ]
            ro = rel_ori[ i, 0, :, :, : ]
            pos = torch.bmm( pos, ro.permute(0,2,1) ) # recover orientation of local regions
            pos += rp.unsqueeze(1) # recover position of local regions
            pc[ :, :, 0:3 ] = pos
            if( C == 6 ): # when each 3D point has a normal vector
                ori = pc[ :, :, 3:6 ]
                ori = torch.bmm( ori, ro.permute(0,2,1) )
                pc[ :, :, 3:6 ] = ori
            pc = torch.reshape( pc, [ G*S, -1 ] )
            out_filepath = "./out/" + str( i ) + "_recovered.xyz"
            np.savetxt( out_filepath, pc.to('cpu').detach().numpy().copy() )
            recovered_shapes.append( pc.unsqueeze(0) )

        recovered_shapes = torch.cat( recovered_shapes, dim=0 )
        return recovered_shapes

    def forward(self, pts, return_tokens=False ):

        # Tokenize
        neighborhood, center, nn_idx = self.group_divider( pts )
        neighborhood_rot, lrf = self.lrf_estimator( neighborhood, center )
        rel_pos, rel_ori = self.relposori_encoder( center, lrf )
        tokens = self.lps_encoder( neighborhood_rot )  #  B G C
        B, G, _ = tokens.shape
        # self.recover_input_shapes( pts, neighborhood_rot, rel_pos, rel_ori )
        # quit()

        # Compute relative position/orientation embedding
        rel_ori = rel_ori.reshape( B, G, G, 9 )
        rel = torch.cat( [ rel_pos, rel_ori ], dim=3 )

        # Encode tokens
        tokens = self.blocks( tokens, rel, self.rel_embedder )
        # tokens = self.norm( tokens )

        if( return_tokens ):
            return tokens, center, neighborhood_rot, nn_idx, lrf

        # Concatenate outputs from all the transformer layers
        tokens = torch.cat( tokens, dim=2 )

        # Compute global features and classify them
        global_feats = tokens.mean(1)
        ret = self.cls_head_finetune( global_feats )
        return ret
