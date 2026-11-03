# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import svraster_cuda

from src.utils.image_utils import resize_rendering
from src.utils.ide_utils import generate_ide_fn_pytorch
class SVRenderer:

    def freeze_vox_geo(self):
        '''
        Freeze grid points parameter and pre-gather them to each voxel.
        '''
        with torch.no_grad():
            self.frozen_vox_geo = svraster_cuda.renderer.GatherGeoParams.apply(
                self.vox_key,
                torch.arange(self.num_voxels, device="cuda"),
                self._geo_grid_pts
            )
        self._geo_grid_pts.requires_grad = False

    def unfreeze_vox_geo(self):
        '''
        Unfreeze grid points parameter.
        '''
        del self.frozen_vox_geo
        self._geo_grid_pts.requires_grad = True

    def vox_fn(self, idx, cam_pos, color_mode=None, viewdir=None):
        '''
        Per-frame voxel property processing. Two important operations:
        1. Gather grid points parameter into each voxel.
        2. Compute view-dependent color of each voxel.

        Input:
            @idx        Indices for active voxel for current frame.
            @cam_pos    Camera position.
        Output:
            @vox_params A dictionary of the pre-process voxel properties.
        '''

        # Gather the density values at the eight corners of each voxel.
        # It defined a trilinear density field.
        # The final tensor are in shape [#vox, 8]
        if hasattr(self, 'frozen_vox_geo'):
            geos = self.frozen_vox_geo
        else:
            geos = svraster_cuda.renderer.GatherGeoParams.apply(
                self.vox_key,
                idx,
                self._geo_grid_pts
            )

        # Compute voxel colors

        if color_mode is None:
            color_mode = "ref"
        ####
        if color_mode == "sh":
            active_sh_degree = self.active_sh_degree
            color_mode = "sh"
        elif color_mode.startswith("sh"):
            active_sh_degree = int(color_mode[2])
            color_mode = "sh"

        if color_mode == "ref":
            # 2. Calcular la normal analítica de cada vóxel visible [M, 3]
            nx = (geos[:, [4, 5, 6, 7]].sum(1) - geos[:, [0, 1, 2, 3]].sum(1))
            ny = (geos[:, [2, 3, 6, 7]].sum(1) - geos[:, [0, 1, 4, 5]].sum(1))
            nz = (geos[:, [1, 3, 5, 7]].sum(1) - geos[:, [0, 2, 4, 6]].sum(1))
            n = torch.stack([nx, ny, nz], dim=-1)
            n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)

            # 3. Vector de visión de la cámara hacia el vóxel
            v = cam_pos - self.vox_center[idx]
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

            # 4. Rayo reflejado r = 2(n · v)n - v
            dot = (n * v).sum(dim=-1, keepdim=True)
            r = 2 * dot * n - v

            # 5. Evaluar Integrated Directional Encoding (IDE) con la rugosidad
            ide_features = self._ide_fn(r, self.roughness[idx])

            # 6. Evaluar componente especular con un Tiny MLP o Environment Map
            dot = n*v
            c_specular = self._specular_mlp(ide_features, self._bottleneck[idx], dot)

            # 7. Componer color por vóxel: Difuso + Tinte * Especular
            rgbs_active = self.albedo[idx] + self.specular_tint[idx] * c_specular

            # Asignar al tensor global rgbs [N, 3]
            rgbs = torch.zeros([self.num_voxels, 3], device="cuda")
            rgbs[idx] = rgbs_active

        elif color_mode == "rand":
            rgbs = torch.rand([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        elif color_mode == "dontcare":
            rgbs = torch.empty([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        else:
            raise NotImplementedError

        # Pack everything
        vox_params = {
            'geos': geos,
            'rgbs': rgbs,
            'subdiv_p': self._subdiv_p, # Dummy param to record subdivision priority
        }
        if vox_params['subdiv_p'] is None:
            vox_params['subdiv_p'] = torch.ones([self.num_voxels, 1], device="cuda")

        return vox_params

    def render(
            self,
            camera,
            color_mode=None,
            track_max_w=False,
            ss=None,
            output_depth=False,
            output_normal=False,
            output_T=False,
            rand_bg=False,
            use_auto_exposure=False,
            **other_opt):

        ###################################
        # Pre-processing
        ###################################
        if ss is None:
            ss = self.ss
        w_src, h_src = camera.image_width, camera.image_height
        w, h = round(w_src * ss), round(h_src * ss)
        w_ss, h_ss = w / w_src, h / h_src
        if ss != 1.0 and 'gt_color' in other_opt:
            other_opt['gt_color'] = resize_rendering(other_opt['gt_color'], size=(h, w))

        n_samp_per_vox = other_opt.pop('n_samp_per_vox', self.n_samp_per_vox)

        ###################################
        # Call low-level rasterization API
        ###################################
        raster_settings = svraster_cuda.renderer.RasterSettings(
            color_mode=color_mode,
            n_samp_per_vox=n_samp_per_vox,
            image_width=w,
            image_height=h,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            cx=camera.cx * w_ss,
            cy=camera.cy * h_ss,
            w2c_matrix=camera.w2c,
            c2w_matrix=camera.c2w,
            bg_color=float(self.white_background),
            near=camera.near,
            need_depth=output_depth,
            need_normal=output_normal,
            track_max_w=track_max_w,
            **other_opt)
        color, depth, normal, T, max_w = svraster_cuda.renderer.rasterize_voxels(
            raster_settings,
            self.octpath,
            self.vox_center,
            self.vox_size,
            self.vox_fn)

        ###################################
        # Post-processing and pack output
        ###################################
        if rand_bg:
            color = color + T * torch.rand_like(color, requires_grad=False)
        elif not self.white_background and not self.black_background:
            color = color + T * color.mean((1,2), keepdim=True)

        if use_auto_exposure:
            color = camera.auto_exposure_apply(color)

        render_pkg = {
            'color': color,
            'depth': depth if output_depth else None,
            'normal': normal if output_normal else None,
            'T': T if output_T else None,
            'max_w': max_w,
        }

        for k in ['color', 'depth', 'normal', 'T']:
            render_pkg[f'raw_{k}'] = render_pkg[k]

            # Post process super-sampling
            if render_pkg[k] is not None and render_pkg[k].shape[-2:] != (h_src, w_src):
                render_pkg[k] = resize_rendering(render_pkg[k], size=(h_src, w_src))

        # Clip intensity
        render_pkg['color'] = render_pkg['color'].clamp(0, 1)

        return render_pkg
