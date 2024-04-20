import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HandcraftedFPFHExtractor( nn.Module ) :
    def __init__( self, num_bins ) :
        super( HandcraftedFPFHExtractor, self ).__init__()
        self.num_bins = num_bins
        self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins

    def forward( self, input ) :
        # input : an input batch of oriented 3D point sets [B, P, 6]

        B, P, _ = input.shape # B: batch size, P: number of 3D point per shape

        pos = input[ :, :, 0:3 ]
        ori = input[ :, :, 3:6 ]

        pi = pos.reshape(B,P,1,3)
        pj = pos.reshape(B,1,P,3)
        ni = ori.reshape(B,P,1,3)
        nj = ori.reshape(B,1,P,3)

        delta = pj - pi
        distance = torch.sqrt( torch.sum( torch.square( delta ), dim=3, keepdim=True ) )

        v = torch.cross( delta, ni.repeat(1,1,P,1), dim=3 )
        v = v / ( torch.linalg.norm( v, dim=3, keepdim=True ) + 1e-6 )
        w = torch.cross( ni.repeat(1,1,P,1), v, dim=3 )
        w = w / ( torch.linalg.norm( w, dim=3, keepdim=True ) + 1e-6 )

        alpha = torch.sum( v * nj, dim=3, keepdim=True )
        phi = torch.sum( ni * delta, dim=3, keepdim=True ) / ( distance + 1e-6 )

        w_nj = torch.sum( w * nj, dim=3, keepdim=True )
        u_nj = torch.sum( ni * nj, dim=3, keepdim=True )
        theta = torch.atan2( w_nj, u_nj ) / np.pi

        # remove point pair features computed from two identical 3D points
        alpha = alpha[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)
        phi = phi[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)
        theta = theta[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)

        # scale to [0,1]
        alpha = ( alpha + 1.0 ) / 2.0
        phi = ( phi + 1.0 ) / 2.0
        theta = ( theta + 1.0 ) / 2.0

        # compute indices for histogramming
        alpha_idx = ( alpha * self.num_bins ).to(torch.int64)
        phi_idx = ( phi * self.num_bins ).to(torch.int64)
        theta_idx = ( theta * self.num_bins ).to(torch.int64)
        alpha_idx[ alpha_idx == self.num_bins ] = self.num_bins - 1
        phi_idx[ phi_idx == self.num_bins ] = self.num_bins - 1
        theta_idx[ theta_idx == self.num_bins ] = self.num_bins - 1

        fpfh_idx = ( ( alpha_idx * self.num_bins ) + phi_idx ) * self.num_bins + theta_idx
        fpfh_idx = fpfh_idx.reshape(B, P*(P-1))

        # compute histograms
        # in this implementation, weighting by distance is ommited for simplicity
        freq = torch.zeros( ( B, self.num_dim_feat ), dtype=torch.float32, device=input.device )
        ones = torch.ones_like( fpfh_idx, dtype=torch.float32 )
        freq = freq.scatter_add( 1, fpfh_idx, ones ) # count population
        freq = freq / (P*(P-1))

        return freq

    def get_outdim( self ) :
        return self.num_dim_feat


# num_bins = 3
#
# B = 2
# P = 4
# C = 6
# pc = torch.rand( B, P, C ).cuda()
# pc[:,:,3:6] = pc[:,:,3:6] / torch.linalg.norm( pc[:,:,3:6], axis=2, keepdim=True )
#
# pos = pc[:,:,0:3]
# ori = pc[:,:,3:6]
#
# pi = pos.reshape(B,P,1,3)
# pj = pos.reshape(B,1,P,3)
# ni = ori.reshape(B,P,1,3)
# nj = ori.reshape(B,1,P,3)
#
# delta = pj - pi
# distance = torch.sqrt( torch.sum( torch.square( delta ), dim=3, keepdim=True ) )
#
# v = torch.cross( delta, ni.repeat(1,1,P,1), dim=3 )
# v = v / ( torch.linalg.norm( v, dim=3, keepdim=True ) + 1e-6 )
# w = torch.cross( ni.repeat(1,1,P,1), v, dim=3 )
# w = w / ( torch.linalg.norm( w, dim=3, keepdim=True ) + 1e-6 )
#
# alpha = torch.sum( v * nj, dim=3, keepdim=True )
# phi = torch.sum( ni * delta, dim=3, keepdim=True ) / ( distance + 1e-6 )
#
# w_nj = torch.sum( w * nj, dim=3, keepdim=True )
# u_nj = torch.sum( ni * nj, dim=3, keepdim=True )
# theta = torch.atan2( w_nj, u_nj ) / np.pi
#
# # remove point pair features computed from two identical 3D points
# alpha = alpha[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)
# phi = phi[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)
# theta = theta[ torch.eye(P).unsqueeze(0).repeat(B,1,1) == 0 ].reshape(B, P, P-1, 1)
#
# # scale to [0,1]
# alpha = ( alpha + 1.0 ) / 2.0
# phi = ( phi + 1.0 ) / 2.0
# theta = ( theta + 1.0 ) / 2.0
#
# # compute indices for histogramming
# alpha_idx = ( alpha * num_bins ).to(torch.int64)
# phi_idx = ( phi * num_bins ).to(torch.int64)
# theta_idx = ( theta * num_bins ).to(torch.int64)
# alpha_idx[ alpha_idx == num_bins ] = num_bins - 1
# phi_idx[ phi_idx == num_bins ] = num_bins - 1
# theta_idx[ theta_idx == num_bins ] = num_bins - 1
#
# fpfh_idx = ( ( alpha_idx * num_bins ) + phi_idx ) * num_bins + theta_idx
# fpfh_idx = fpfh_idx.reshape(B, P*(P-1))
#
# # compute histograms
# # in this implementation, weighting by distance is ommited for simplicity
# freq = torch.zeros( ( B, num_bins*num_bins*num_bins ), dtype=torch.float32, device=pc.device )
# ones = torch.ones_like( fpfh_idx, dtype=torch.float32 )
# freq = freq.scatter_add( 1, fpfh_idx, ones ) # count population
# freq = freq / (P*(P-1))
