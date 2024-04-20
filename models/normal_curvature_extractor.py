import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.common.workaround import symeig3x3

class NormalCurvatureExtractor( nn.Module ) :
    def __init__( self ) :
        super( NormalCurvatureExtractor, self ).__init__()
        self.num_dim_feat = 4 # 3 for normal vector and 1 for surface variation

    def forward( self, input ) :
        # input : an input batch of oriented 3D point sets [B, P, 6]

        B, P, _ = input.shape # B: batch size, P: number of 3D point per shape

        pos = input[ :, :, 0:3 ]
        ori = input[ :, :, 3:6 ]

        # ===== normal vector of center point =====
        normal = ori[ :, 0, : ] # for each LRF, 0-th point corresponds to center point

        # ===== curvature (surface variation) =====
        pos = pos.unsqueeze(0)
        # generate covariance matrices
        scaled_pos = 100.0 * pos # for numerical stability
        covs = torch.einsum( "bijk,bijl->bikl", scaled_pos, scaled_pos )

        # PCA
        eigvalue, _ = symeig3x3( covs )
        eigvalue = eigvalue.reshape(B,3)
        surface_variation = eigvalue[:,0].unsqueeze(1) / torch.sum( eigvalue, dim=1, keepdim=True )

        ret = torch.cat( [ normal, surface_variation ], dim=1 )
        return ret

    def get_outdim( self ) :
        return self.num_dim_feat
