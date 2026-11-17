import socket
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_vertex_joint import joint_scene
from src.config import cfg
from viz_vertex_fourier3 import load_viewer,VertexViewer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_joint_loader_and_headless_viewer(tmp_path,monkeypatch):
    import viz_vertex_fourier3 as viz
    model,camera=joint_scene();model.activate_vertex()
    config=cfg.clone();config.merge_from_file('cfg/synthetic_nerf.yaml');config.model.ss=1.0
    model.joint_config=config.dump()
    path=tmp_path/'model.pt';model.save(str(path))
    class Pack:
        def __init__(self,**kwargs): assert kwargs['camera_params_only']
        def get_train_cameras(self): return [camera]
        def get_test_cameras(self): return []
    monkeypatch.setattr(viz,'DataPack',Pack)
    config,loaded,module,pack,is_joint=load_viewer(path)
    assert is_joint and module is not None
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));config.port=sock.getsockname()[1]
    viewer=VertexViewer(config,loaded,module,pack,True)
    try:
        viewer.width_slider.value=64
        handle=SimpleNamespace(wxyz=np.array([1.,0.,0.,0.]),position=np.array([0.,0.,-3.]),aspect=1.)
        for output in ('rgb','alpha','dmean','n'):
            viewer.output_dropdown.value=output
            image,seconds=viewer.render_viser_camera(handle)
            assert viewer.status.value=='Ready'
            assert image.shape==(64,64,3)
            assert seconds>0
        viewer.appearance.value='SH0'
        image,_=viewer.render_viser_camera(handle)
        assert viewer.status.value=='Ready'
        assert 'SH3' not in viewer.appearance.options
    finally:
        viewer.server.stop()
