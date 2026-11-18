import torch
from . import _C


class RasterizeRBF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        subdiv_p,
        rbf_eta,
        rbf_keys,
    ):

        need_distortion = raster_settings.lambda_dist > 0

        args = (
            raster_settings.n_samp_per_vox,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.bg_color,
            raster_settings.need_depth,
            need_distortion,
            raster_settings.need_normal,
            raster_settings.track_max_w,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,

            geomBuffer,

            raster_settings.debug,
        )

        num_rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w = _C.rasterize_rbf(*(args[:19] + (rbf_eta, rbf_keys) + args[19:]))

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            octree_paths, vox_centers, vox_lengths,
            geos, rgbs,
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal, rbf_eta, rbf_keys)
        ctx.mark_non_differentiable(max_w)
        return out_color, out_depth, out_normal, out_T, max_w

    @staticmethod
    def backward(ctx, dL_dout_color, dL_dout_depth, dL_dout_normal, dL_dout_T, dL_dmax_w):
        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        octree_paths, vox_centers, vox_lengths, \
            geos, rgbs, \
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal, rbf_eta, rbf_keys = ctx.saved_tensors

        args = (
            num_rendered,
            raster_settings.n_samp_per_vox,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.bg_color,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,

            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,

            dL_dout_color,
            dL_dout_depth,
            dL_dout_normal,
            dL_dout_T,

            raster_settings.lambda_R_concen,
            raster_settings.gt_color,
            raster_settings.lambda_ascending,
            raster_settings.lambda_dist,
            raster_settings.need_depth,
            raster_settings.need_normal,
            out_depth,
            out_normal,

            raster_settings.debug,
        )

        dL_dgeos, dL_drgbs, subdiv_p_bw, dL_deta = _C.rasterize_rbf_backward(*(args[:16] + (rbf_eta,rbf_keys) + args[16:]))

        grads = (
            None, # => raster_settings
            None, # => geomBuffer
            None, # => octree_paths
            None, # => vox_centers
            None, # => vox_lengths
            dL_dgeos, # => geos
            dL_drgbs, # => rgbs
            subdiv_p_bw, # => subdivision priority
        )

        return grads + (dL_deta, None)

