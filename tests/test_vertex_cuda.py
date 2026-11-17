import math
import pytest
import torch
from src.vertex_appearance import VertexRGB, VertexAppearance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def scene():
    from src.sparse_voxel_model import SparseVoxelModel
    from src.utils.octree_utils import gen_octpath_dense
    from src.cameras import MiniCam
    m = SparseVoxelModel(ss=1, white_background=True)
    m.scene_center = torch.zeros(3,device='cuda')
    m.scene_extent = torch.tensor(2.,device='cuda')
    m.inside_extent = m.scene_extent
    m.octpath,m.octlevel = gen_octpath_dense(0,1)
    m._geo_grid_pts = torch.full((m.num_grid_pts,1), -1.,device='cuda')
    m._sh0 = torch.zeros((m.num_voxels,3),device='cuda')
    m._shs = torch.zeros((m.num_voxels,15,3),device='cuda')
    m._subdiv_p = torch.ones((m.num_voxels,1),device='cuda')
    c2w = torch.eye(4)
    c2w[2,3] = -3
    cam = MiniCam(c2w,math.pi/3,math.pi/3,16,16)
    m.freeze_vox_geo()
    return m,cam


def test_zero_regression_and_geometry_invariance():
    m,c = scene()
    a = VertexRGB(m.num_grid_pts).cuda()
    opts = dict(output_depth=True,output_T=True,output_normal=True)
    reference = m.render(c,color_mode='sh0',**opts)
    zero = m.render_vertex(c,a,**opts)
    assert (reference['T']<1).any()
    torch.testing.assert_close(zero['color'],reference['color'],atol=1e-5,rtol=0)
    with torch.no_grad():
        a.vertex_rgb.uniform_(-0.2,0.2)
    other = m.render_vertex(c,a,**opts)
    for key in ('depth','T','normal'):
        torch.testing.assert_close(other[key],zero[key],atol=1e-5,rtol=0)
    with pytest.raises(ValueError):
        m.render_vertex(c,a,n_samp_per_vox=2)


def test_backward_finite_difference():
    m,c = scene()
    a = VertexRGB(m.num_grid_pts).cuda()
    a.vertex_rgb.data.uniform_(-0.1,0.1)
    loss = m.render_vertex(c,a)['color'].mean()
    loss.backward()
    analytical = a.vertex_rgb.grad.clone()
    assert analytical.abs().sum()>0
    for vertex in (0,5,13):
        eps = 1e-3
        with torch.no_grad():
            a.vertex_rgb[vertex,0] += eps
            plus=m.render_vertex(c,a)['color'].mean()
            a.vertex_rgb[vertex,0] -= 2*eps
            minus=m.render_vertex(c,a)['color'].mean()
            a.vertex_rgb[vertex,0] += eps
        torch.testing.assert_close(analytical[vertex,0],(plus-minus)/(2*eps),atol=1e-4,rtol=1e-2)


def test_decoder_overfit():
    m,c=scene()
    a=VertexAppearance(m.num_grid_pts).cuda()
    opt=torch.optim.Adam(a.parameters(),lr=0.01)
    losses=[]
    for _ in range(8):
        opt.zero_grad()
        loss=(m.render_vertex(c,a)['color']-0.7).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        opt.step()
        losses.append(loss.item())
    assert losses[-1]<losses[0]
