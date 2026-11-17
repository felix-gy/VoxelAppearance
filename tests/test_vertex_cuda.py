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


def test_constant_residual_matches_constant_sh_color():
    m,c=scene()
    a=VertexRGB(m.num_grid_pts).cuda()
    a.vertex_rgb.data.fill_(0.1)
    actual=m.render_vertex(c,a)['color']
    m._sh0.fill_(0.1/0.28209479177387814)
    expected=m.render(c,color_mode='sh0')['color']
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)


def test_color_clamp_has_zero_gradient():
    m,c=scene()
    a=VertexRGB(m.num_grid_pts).cuda()
    a.vertex_rgb.data.fill_(-1)
    m.render_vertex(c,a)['color'].sum().backward()
    assert a.vertex_rgb.grad.count_nonzero()==0


def test_checkpoint_preserves_render(tmp_path):
    from src.vertex_appearance.checkpoint import save_checkpoint,load_checkpoint,layout_fingerprint
    m,c=scene()
    a=VertexAppearance(m.num_grid_pts).cuda()
    a.decoder[-1].weight.data.normal_(0,0.01)
    opt=torch.optim.Adam(a.parameters())
    generator=torch.Generator().manual_seed(3)
    identity={'layout':layout_fingerprint(m)}
    path=tmp_path/'state.pt'
    expected=m.render_vertex(c,a)['color'].detach()
    save_checkpoint(path,a,opt,1,{'mode':'vertex_trilinear_fourier3'},identity,generator)
    other=VertexAppearance(m.num_grid_pts).cuda()
    load_checkpoint(path,other,identity)
    torch.testing.assert_close(m.render_vertex(c,other)['color'],expected,atol=1e-5,rtol=0)


def test_empty_frustum_and_partial_visibility():
    m,c=scene()
    a=VertexRGB(m.num_grid_pts).cuda()
    # Rotate around y: all geometry lies behind this camera.
    c.c2w[:3,:3]=torch.diag(torch.tensor([-1.,1.,-1.],device='cuda'))
    c.w2c=c.c2w.inverse()
    out=m.render_vertex(c,a,output_T=True)
    torch.testing.assert_close(out['T'],torch.ones_like(out['T']))
    out['color'].sum().backward()
    assert a.vertex_rgb.grad.count_nonzero()==0
    c.c2w[:3,:3]=torch.eye(3,device='cuda')
    c.c2w[0,3]=1.5
    c.w2c=c.c2w.inverse()
    ref=m.render(c,color_mode='sh0')['color']
    torch.testing.assert_close(m.render_vertex(c,a)['color'],ref,atol=1e-5,rtol=0)


def test_layout_matches_kernel_corner_order():
    m,c=scene()
    bits=torch.tensor([[k>>2,(k>>1)&1,k&1] for k in range(8)],device='cuda',dtype=torch.float32)
    expected=m.vox_center[:,None,:]+(bits[None,:,:]-0.5)*m.vox_size[:,None,:]
    torch.testing.assert_close(m.grid_pts_xyz[m.vox_key],expected,atol=1e-6,rtol=0)


def test_two_view_overfit():
    m,c=scene()
    c2=c.clone_mini()
    c2.c2w[0,3]=0.3
    c2.w2c=c2.c2w.inverse()
    a=VertexAppearance(m.num_grid_pts).cuda()
    opt=torch.optim.Adam(a.parameters(),lr=0.01)
    losses=[]
    for _ in range(12):
        opt.zero_grad()
        loss=sum((m.render_vertex(cam,a)['color']-target).square().mean()
                 for cam,target in ((c,0.7),(c2,0.5)))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1]<losses[0]
    assert not torch.allclose(a(m.grid_pts_xyz,c.position),a(m.grid_pts_xyz,c2.position))


def test_wrapper_never_trains_base_sh():
    m,c=scene()
    m._sh0.requires_grad_(True)
    m._shs.requires_grad_(True)
    a=VertexRGB(m.num_grid_pts).cuda()
    m.render_vertex(c,a)['color'].mean().backward()
    assert m._sh0.grad is None and m._shs.grad is None
    assert a.vertex_rgb.grad.abs().sum()>0


def test_layout_validation_rejects_invalid_keys():
    from src.vertex_appearance.layout import validate_layout
    m,c=scene()
    validate_layout(m)
    m._vox_key=m.vox_key.clone()
    m._vox_key[0,0]=m.num_grid_pts
    with pytest.raises(ValueError,match='out of range'):
        validate_layout(m)
