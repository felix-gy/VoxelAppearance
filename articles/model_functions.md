# Model functions

We cover the functions to create a model instance in [model_creation_tutor.md](./model_creation_tutor.md). We describe the other functions in the following.

## Properties
- `n_samp_per_vox`: Number of samples per voxel when rendering.
- `ss`: Super-sampling scale.
    - We render higher-resolution image (`[H * ss, W * ss]`) and apply anti-aliasing downsampling.
- `white_background`: Indicate if the background is white.
- `black_background`: Indicate if the background is black.
    - The background will be the averaged color if neither black or white is set.
- `ide_degree`: Fixed directional encoding degree, from 1 to 5.
- `bottleneck_dim`: Number of learned latent features per voxel.
- `hidden_dim`: Width of the shared specular MLP.
- `normal_eps`: Threshold below which a density gradient has no reliable normal.

## Derived properties
- `num_voxels`: Number of voxels.
- `num_grid_pts`: Number of grid points.
    - Recap that a voxel has 8 corner grid points. A grid point can be shared by adjacent voxels. This is the number of unique grid points.
- `scene_min`: Minimum coordinate of entire scene.
- `scene_max`: Maximum coordinate of entire scene.
- `inside_min`: Minimum coordinate of the main foreground region.
    - It's valid when the model is created from `model_init` with `outside_level > 0` which preserves Octree level outside the main foreground bounding box.
- `inside_max`: Maximum coordinate of the main foreground region.
- `inside_mask`: A mask indicating if a voxel is in `inside_min` and `inside_max`.
- `subdivision_priority`: The model automatically tracks and accumulates subdivision priority during rendering backward pass.
    - Larger value means higher priority.
    - Reset by `reset_subdivision_priority()`.

The following is the properties that lazily computed at the first time you access them. It automatically recompute when it detect the voxels allocation is updated (e.g., after pruning or subdivision).
- `vox_center`: Voxel center position in the world space.
- `vox_size`: Voxel size.
- `vox_key`: Index to the unique grid points. It's in shape of `[num_voxels, 8]`.
- `grid_pts_xyz`: The world-space position of the unique grid points.

## Parameters
- `_albedo`, `_specular_tint`: Raw RGB parameters, each `[num_voxels, 3]`.
- `_roughness`: Raw smoothing parameter, `[num_voxels, 1]`.
- `_bottleneck`: Latent features, `[num_voxels, bottleneck_dim]`.
- `_specular_mlp`: One shared network. Its weights are not copied during subdivision.
- `albedo`, `specular_tint`, `roughness`: Activated properties, retaining autograd.
- `_geo_grid_pts`: The density of grid points. The shape is `[num_grid_pts, 1]`.
    - When rendering, it's gathered into `[num_voxels, 8]` as voxel trilinear density field for the CUDA to render.

## Core functions
- `render_pkg = render(camera, track_max_w=False, output_depth=False, output_normal=False, output_T=False)`
    - Rendering a view.
    - `track_max_w` whether to track the maximum blending weigth of each voxel. Access by `render_pkg['max_w']`.
    - `output_depth` whether to render depth. Access by `render_pkg['depth']` or `render_pkg['raw_depth']`.
    - `output_normal` whether to render normal. Access by `render_pkg['normal']` or `render_pkg['raw_normal']`.
    - `output_T` whether to output transmittance. Access by `render_pkg['T']` or `render_pkg['raw_T']`.
    - `color_mode` accepts `ref` (default), `albedo`, `rand`, or `dontcare`.
    - `color` is clipped to `[0, 1]`; `color_unclamped` has the target image resolution and preserves gradients outside that interval for training.
    - The outputs with `raw_` prefix are the results without anti-aliasing downsampling.
    - The depth and normal is not normalized by alpha.
    - There output depth is in shape `[3, H, W]` for mean depth, distortion cache, median depth. Only the mean depth support backpropagation.
- `pruning(mask)`
    - Remove voxels indicating by the given mask.
- `subdividing(mask)`
    - Subdivde voxels into their eight octans indicating by the given mask. The source parent voxels are removed after subdivision.

## Useful functions
- `compute_training_stat(camera_lst)`
    - Compute the per-voxel statistic from the given cameras, including `max_w` for maximum blending weight, `min_samp_interval` for the inverse of maximum sampling rates, and `view_cnt` for visibile camera count.
- `reset_albedo_from_cameras(camera_lst)`
    - Initialize albedo from fused camera colors; shared specular weights remain intact.
- `apply_tv_on_density_field(lambda_tv_density)`
    - Add the gradient of total variation loss to the `_geo_grid_pts` parameter.
- `save(path, quantize=False)`
    - Save the model to the given path. You can optionally apply 8-bit quantization to the parameters while keeping shared network weights in full precision. Quantization is lossy.
- `load(path)`
    - Load checkpoint, including the saved appearance architecture and network weights. Legacy SH checkpoints are approximate warm starts, as described in [ref_appearance.md](./ref_appearance.md).
- `load_iteration(model_path, iteration=-1)`
    - Load checkpoint from a model output path with the given iteration. The default load the latest iteration.