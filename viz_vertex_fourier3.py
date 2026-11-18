"""Interactive viewer for joint checkpoints and frozen VertexAppearance v2."""
import argparse
import threading
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from yacs.config import CfgNode
from src.config import cfg as default_config
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.vertex_appearance.joint import JointVoxelModel
from src.vertex_appearance.decoder import VertexAppearance, VertexRGB
from src.vertex_appearance.checkpoint import file_digest, layout_fingerprint, load_checkpoint
from viz import SVRasterViewer


def load_viewer(checkpoint, geometry_checkpoint=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = default_config.clone()
    module = None
    is_joint = 'vertex_joint' in state
    if is_joint:
        config.merge_from_other_cfg(CfgNode.load_cfg(state['vertex_joint']['config']))
        model_class = JointVoxelModel
        if state["vertex_joint"].get("kernel") == "vertex_rbf_iq":
            from src.vertex_appearance.rbf import RBFJointModel
            model_class = RBFJointModel
        model = model_class(**dict(config.model))
        model.load(checkpoint)
        module = model.vertex_module
    else:
        if state.get('schema_version') != 2:
            raise ValueError('Expected joint checkpoint or frozen appearance schema v2')
        appearance = state['config']
        base = appearance.get('resolved_base') or yaml.safe_load(Path(appearance['base_config']).read_text())
        config.merge_from_other_cfg(CfgNode(base))
        model = SparseVoxelModel(**dict(config.model))
        geometry = geometry_checkpoint or appearance['geometry_checkpoint']
        model.load(geometry)
        module = (VertexAppearance if state['mode'] == 'vertex_trilinear_fourier3' else VertexRGB)(model.num_grid_pts).cuda()
        identity = dict(checkpoint_sha256=file_digest(geometry), layout=layout_fingerprint(model),
                        P=model.num_grid_pts,V=model.num_voxels)
        load_checkpoint(checkpoint,module,identity)
    model.freeze_vox_geo()
    if module is not None:
        module.eval()
    data = dict(config.data)
    data['use_test'] = data.pop('eval')
    data['alpha_is_white'] = config.model.white_background
    data['camera_params_only'] = True
    return config,model,module,DataPack(**data),is_joint


class VertexViewer(SVRasterViewer):
    def __init__(self, config, model, module, cameras, is_joint):
        self.render_lock = threading.Lock()
        self.last_image = None
        self.vertex_module = module
        super().__init__(config,model=model,camera_pack=cameras)
        self.active_sh_degree_slider.visible = False
        self.vertex_label = ('Vertex-MLP + RBF-IQ' if hasattr(module,'rbf_log_scales') else ('Vertex-RGB' if isinstance(module, VertexRGB) else 'Vertex-MLP'))
        modes = ['SH0'] if module is None else [self.vertex_label,'SH0']
        if hasattr(module,'rbf_log_scales'):
            modes.insert(1,'Vertex-MLP trilineal')
        if not is_joint and model.shs.shape[1] >= 15:
            modes.append('SH3')
        self.appearance = self.server.gui.add_dropdown('Appearance',options=modes,initial_value=modes[0])
        self.status = self.server.gui.add_text('Render status',initial_value='Ready',disabled=True)

    def render_model(self, camera, options):
        options = dict(options)
        # Output selection always requests the buffers it consumes.
        selected = self.output_dropdown.value
        if selected in ('dmean','dmed','dmean2n','dmed2n'):
            options['output_depth'] = True
        if selected == 'n':
            options['output_normal'] = True
        options.pop('color_mode',None)
        if self.appearance.value == 'Vertex-MLP trilineal':
            return self.voxel_model.render_vertex(camera,self.vertex_module,interpolation='trilinear',**options)
        if self.appearance.value == self.vertex_label:
            return self.voxel_model.render_vertex(camera,self.vertex_module,**options)
        options['color_mode'] = 'sh3' if self.appearance.value == 'SH3' else 'sh0'
        return self.voxel_model.render(camera,**options)

    @torch.no_grad()
    def render_viser_camera(self, camera):
        with self.render_lock:
            try:
                result = super().render_viser_camera(camera)
                self.last_image = result[0]
                self.status.value = 'Ready'
                return result
            except (RuntimeError,ValueError,TypeError) as error:
                self.status.value = str(error)
                if self.last_image is None:
                    return np.zeros((64,64,3),dtype=np.uint8),1.0
                return self.last_image,1.0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--geometry-checkpoint')
    parser.add_argument('--port',type=int,default=7007)
    args=parser.parse_args()
    config,model,module,cameras,is_joint=load_viewer(args.checkpoint,args.geometry_checkpoint)
    if not cameras.get_train_cameras():
        raise ValueError('Viewer requires at least one training camera')
    config.port=args.port
    viewer=VertexViewer(config,model,module,cameras,is_joint)
    while True:
        viewer.update()
        time.sleep(0.003)


if __name__ == '__main__':
    main()
