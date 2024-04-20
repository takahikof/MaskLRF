import numpy as np
import torch
import random
from pytorch3d.ops import estimate_pointcloud_normals
from pointmixup import point_mixup

def uniform_random_rotation():
    # https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/
    """Sample a random rotation in 3D, with a distribution uniform over the sphere.
    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x, about the mean coordinate of x.
    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """
    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R
    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()
    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    return M

class PointcloudRotateSO3(object):
    def __call__(self, pc): # pc: [batch, num_points, 6]
        bsize = pc.size()[0]
        C = pc.size()[2]
        for i in range(bsize):
            R = uniform_random_rotation()
            R = torch.from_numpy( R ).float().to( pc.device )
            pos = pc[ i, :, 0:3 ]
            pos = torch.mm( pos, R )
            pc[ i, :, 0:3 ] = pos
            if( C == 6 ):
                ori = pc[ i, :, 3:6 ]
                ori = torch.mm( ori, R )
                pc[ i, :, 3:6 ] = ori
        return pc

class PointcloudScaleAnisotropic(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc): # pc: [batch, num_points, 6]
        # anisotropic scaling (orthogonal axes for scaling are chosen randomly)
        bsize = pc.size()[0]
        for i in range(bsize):
            R = uniform_random_rotation()
            R = torch.from_numpy( R ).float().to( pc.device )
            pos = pc[ i, :, 0:3 ]
            pos = torch.mm( pos, R )
            scaling_factors = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pos = torch.mul( pos, torch.from_numpy(scaling_factors).float().to( pc.device ) )
            pos = torch.mm( pos, torch.transpose( R, 0, 1 ) )
            pc[ i, :, 0:3 ] = pos

        return pc

class PointcloudScaleUniform(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc): # pc: [batch, num_points, 6]
        # uniform scaling
        bsize = pc.size()[0]
        for i in range(bsize):
            scaling_factor = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[1,1])
            pos = pc[ i, :, 0:3 ]
            pos = torch.mul( pos, torch.from_numpy(scaling_factor).float().to( pc.device ) )
            pc[ i, :, 0:3 ] = pos

        return pc

class PointcloudRandomCrop(object):
    def __init__(self, scale_min=0.6, scale_max=1.0):
        # assert scale_min >= 0.5 and scale_min <= 1.0
        # assert scale_max >= scale_min and scale_max <= 1.0
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, pc):
        bsize = pc.size()[0]
        P = pc.size()[1]
        C = pc.size()[2]
        for i in range(bsize):
            ops = pc[i]
            pos = ops[:,0:3]

            # randomly choose scale of cropped 3D point set
            scale = np.random.uniform( low=self.scale_min, high=self.scale_max )

            # convert the scale to the number of neighboring points
            knn = int( P * scale )

            randint = np.random.randint( 0, P )
            distvec = torch.cdist( pos[ randint ].unsqueeze(0), pos )
            dists, nn_idx = torch.topk( distvec, knn, dim=1, largest=False, sorted=True )

            nn_idx = nn_idx.squeeze()
            nn_idx = nn_idx.unsqueeze(1).repeat(1,C)
            crop = torch.gather( ops, dim=0, index=nn_idx )

            # duplicate points so that the cropped 3D point set has P points
            randidx = torch.randperm( knn )
            nn_idx = nn_idx[ randidx ]
            crop_add = torch.gather( ops, dim=0, index=nn_idx[0:(P-knn)] )

            pc[ i ] = torch.cat( [ crop, crop_add ], dim=0 )

        return pc

class PointcloudMixup(object):
    def __init__(self, alpha=1.0, mode="K" ):
        self.alpha = alpha
        self.mode = mode
    def __call__(self, pc):
        mixup_pointsets, mixup_lambdas, mixup_idxs = point_mixup( pc, mixup_alpha=self.alpha, mixup_mode=self.mode )
        return mixup_pointsets, mixup_lambdas, mixup_idxs

class PointcloudEstimateSurfaceNormals(object):
    def __init__(self):
        return

    def __call__(self, pc): # pc: [batch, num_points, 3]
        # k is 32 when each point set has 1024 points
        # k is 64 when each point set has 2048 points
        k = int( pc.shape[1] * 0.03125 )
        normals = estimate_pointcloud_normals( pc, neighborhood_size=k, disambiguate_directions=True )
        pc = torch.cat( [ pc, normals ], dim=2 )
        return pc # [batch, num_points, 6]
