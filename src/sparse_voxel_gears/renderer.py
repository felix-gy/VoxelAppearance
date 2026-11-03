# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn.functional as F
import svraster_cuda

from src.utils.image_utils import resize_rendering

class SVRenderer:

    def freeze_vox_geo(self):
        '''
        Freeze grid points parameter and pre-gather them to each voxel.
        '''
        if hasattr(self, 'frozen_vox_geo'):
            return
        with torch.no_grad():
            self.frozen_vox_geo = svraster_cuda.renderer.GatherGeoParams.apply(
                self.vox_key,
                torch.arange(self.num_voxels, device=self.octpath.device),
                self._geo_grid_pts
            )

    def unfreeze_vox_geo(self):
        '''
        Unfreeze grid points parameter.
        '''
        if self._geo_grid_pts is None:
            from src.sparse_voxel_gears.adaptive import agg_voxel_into_grid_pts
            self._geo_grid_pts = agg_voxel_into_grid_pts(
                self.num_grid_pts, self.vox_key,
                self.frozen_vox_geo.unsqueeze(-1)).requires_grad_()
        del self.frozen_vox_geo

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

        color_mode = "ref" if color_mode is None else color_mode
        if color_mode == "ref":
            active_rgb = self.evaluate_color(idx, geos[idx], cam_pos, viewdir)
            rgbs = geos.new_zeros((self.num_voxels, 3))
            rgbs[idx] = active_rgb
        elif color_mode == "albedo":
            rgbs = geos.new_zeros((self.num_voxels, 3))
            rgbs[idx] = torch.sigmoid(self._albedo[idx])
        elif color_mode == "rand":
            rgbs = torch.rand(self.num_voxels, 3, device=geos.device, dtype=geos.dtype)
        elif color_mode == "dontcare":
            rgbs = geos.new_zeros((self.num_voxels, 3))
        else:
            raise ValueError(
                f"Unknown color mode {color_mode!r}; use ref, albedo, rand or dontcare. "
                "Per-voxel SH colors have been replaced by REF appearance.")

        # Pack everything
        vox_params = {
            'geos': geos,
            'rgbs': rgbs,
            'subdiv_p': self._subdiv_p, # Dummy param to record subdivision priority
        }
        if vox_params['subdiv_p'] is None:
            vox_params['subdiv_p'] = geos.new_ones((self.num_voxels, 1))

        return vox_params

    def evaluate_color(self, idx, active_geos, cam_pos=None, viewdir=None):
        """Shade selected voxels; active_geos is [len(idx), 8].

        Optional viewdir points from camera to voxel, matching the previous
        interface, and may be [3], [N, 3], or [len(idx), 3]. Colors are learned
        directly in the training images' encoding; no sRGB conversion is added.
        """
        if active_geos.shape != (len(idx), 8):
            raise ValueError("active_geos must have shape [len(idx), 8].")
        if viewdir is None:
            if cam_pos is None:
                raise ValueError("Provide cam_pos or viewdir to evaluate color.")
            v = cam_pos.to(active_geos) - self.vox_center[idx].to(active_geos)
        else:
            if viewdir.shape == (self.num_voxels, 3):
                viewdir = viewdir[idx]
            elif viewdir.shape not in ((3,), (len(idx), 3)):
                raise ValueError("viewdir must have shape [3], [N, 3], or [len(idx), 3].")
            v = -viewdir.to(active_geos).expand(len(idx), 3)
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v = torch.where(v_norm > self.normal_eps,
                        v / v_norm.clamp_min(self.normal_eps),
                        v.new_tensor([0.0, 0.0, 1.0]))

        # Outward normal from the negative density gradient, with x/y/z corner bits
        # 100/010/001. Its common scale cancels under normalization.
        g = active_geos
        gradient = torch.stack([
            g[:, [4, 5, 6, 7]].sum(1) - g[:, [0, 1, 2, 3]].sum(1),
            g[:, [2, 3, 6, 7]].sum(1) - g[:, [0, 1, 4, 5]].sum(1),
            g[:, [1, 3, 5, 7]].sum(1) - g[:, [0, 2, 4, 6]].sum(1),
        ], dim=-1)
        magnitude = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
        normal = -gradient / magnitude.clamp_min(self.normal_eps)
        # A uniform density cell has no defined normal. Use a detached unit
        # fallback and zero geometry gradients there instead of amplifying noise.
        normal = torch.where(magnitude > self.normal_eps, normal, v.detach())
        dot = (normal * v).sum(dim=-1, keepdim=True)
        reflected = F.normalize(2 * dot * normal - v, dim=-1, eps=self.normal_eps)
        roughness = F.softplus(self._roughness[idx]) + self.MIN_ROUGHNESS
        ide_features = self._ide_fn(reflected, roughness)
        specular = self._specular_mlp(ide_features, self._bottleneck[idx], dot)
        return torch.sigmoid(self._albedo[idx]) + torch.sigmoid(self._specular_tint[idx]) * specular

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

        # Photometric training uses this target-resolution image so values above
        # one can still receive a gradient. Keep display/evaluation RGB bounded.
        render_pkg['color_unclamped'] = render_pkg['color']

        # Clip intensity
        render_pkg['color'] = render_pkg['color'].clamp(0, 1)

        return render_pkg
