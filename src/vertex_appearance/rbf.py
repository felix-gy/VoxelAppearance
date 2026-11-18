"""Local anisotropic inverse-quadratic RGB modulation; density is unchanged."""
import torch
from torch import nn
from .decoder import VertexAppearance
from .joint import JointVoxelModel


class RBFAppearance(VertexAppearance):
    def __init__(self, num_vertices):
        super().__init__(num_vertices)
        self.rbf_log_scales = nn.Parameter(torch.zeros(num_vertices,3))

    @torch.no_grad()
    def project_scales(self):
        if not torch.isfinite(self.rbf_log_scales).all():
            raise RuntimeError('Nonfinite RBF log scales')
        self.rbf_log_scales.clamp_(-2,2)


def reference_weights(t, eta):
    bits=t.new_tensor([[k>>2,(k>>1)&1,k&1] for k in range(8)])
    delta=t[...,None,:]-bits
    N=torch.where(bits.bool(),t[...,None,:],1-t[...,None,:]).prod(-1)
    D0=delta.square().sum(-1)
    D=(delta.square()*torch.exp(-2*eta.clamp(-2,2))).sum(-1)
    unnormalized=N*(1+D0)/(1+D)
    return unnormalized/unnormalized.sum(-1,keepdim=True)


class RBFJointModel(JointVoxelModel):
    appearance_kernel="vertex_rbf_iq"
    appearance_spec=dict(version=1,kernel='relative_inverse_quadratic',
                         log_scale_bounds=[-2.0,2.0],coordinates='voxel_local',
                         feature_dim=8,mlp=[26,64,3],fourier_frequencies_pi=[1.0,2.0,4.0])
    def activate_vertex(self):
        if self.vertex_module is not None:
            return False
        with torch.random.fork_rng():
            self.vertex_module=RBFAppearance(self.num_grid_pts).cuda()
        self.active_sh_degree=0
        self._shs.requires_grad_(False)
        self._shs.grad=None
        return True

    @torch.no_grad()
    def _adapt_vertex(self, mask, subdivide):
        if self.vertex_module is None:
            return super()._adapt_vertex(mask,subdivide)
        keys=self.grid_pts_key.clone()
        eta=self.vertex_module.rbf_log_scales.detach()
        super()._adapt_vertex(mask,subdivide)
        both=torch.cat((keys,self.grid_pts_key))
        _,inv=both.unique(dim=0,return_inverse=True)
        mapping=torch.full((int(inv.max())+1,),-1,device=keys.device,dtype=torch.long)
        mapping[inv[:len(keys)]]=torch.arange(len(keys),device=keys.device)
        source=mapping[inv[len(keys):]]
        values=torch.zeros((self.num_grid_pts,3),device=eta.device)
        valid=source>=0
        values[valid]=eta[source[valid]]
        self.vertex_module.rbf_log_scales=nn.Parameter(values)
