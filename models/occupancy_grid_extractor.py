import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OccupancyGridExtractor( nn.Module ) :
    def __init__( self, num_bins ) :
        super( OccupancyGridExtractor, self ).__init__()
        self.num_bins = num_bins
        self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins

    def forward( self, input ) :
        # input : an input batch of oriented 3D point sets [B, P, 6]

        B, P, _ = input.shape # B: batch size, P: number of 3D point per shape

        pos = input[ :, :, 0:3 ]

        # Each point set is "voxelized" by using regular 3D grids.

        # Firstly, normalize scale of token point sets so that each token point set is fit in a unit cube.
        bbox_max, _ = torch.max( torch.abs( pos ), dim=1, keepdim=True )
        bbox_max, _ = torch.max( bbox_max, dim=2, keepdim=True )
        bbox_max = torch.tile( bbox_max, ( 1, 1, 3 ) )
        bbox_min = - bbox_max
        thickness = torch.clamp( bbox_max - bbox_min, min=1e-5 )
        pos_in_cube = ( pos - bbox_min ) / thickness

        # Assign a cell ID to each point (cell ID indicates which cell in a 3D grid a point belongs to.)
        cell_idx = ( pos_in_cube * self.num_bins ).to( torch.int64 )
        cell_idx = torch.clamp( cell_idx, min=0, max=self.num_bins-1 )
        cell_idx = ( cell_idx[ :, :, 0 ] * self.num_bins + cell_idx[ :, :, 1 ] ) * self.num_bins + cell_idx[ :, :, 2 ]

        # frequency of points for each cell
        freq = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins ), dtype=torch.float32, device=input.device )
        ones = torch.ones_like( cell_idx, dtype=torch.float32 )
        freq = freq.scatter_add( 1, cell_idx, ones ) # count population

        grid = ( freq > 0.0 ).float() # binarize

        return grid

    def get_outdim( self ) :
        return self.num_dim_feat
