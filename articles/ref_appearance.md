# REF appearance integration

This fork replaces per-voxel SH coefficients with raw albedo `[N, 3]`, tint
`[N, 3]`, roughness `[N, 1]` and latent features `[N, bottleneck_dim]`. A single
MLP shared by all voxels predicts the specular RGB from reflected-direction
IDE, the latent vector and the scalar normal/view dot product.

```python
model = SparseVoxelModel(ide_degree=5, bottleneck_dim=8, hidden_dim=64)
# Allocate a layout with model_init(), octpath_init(), ijkl_init(), or points_init().
color = model.render(camera)['color']
diffuse_only = model.render(camera, color_mode='albedo')['color']
albedo_i = model.albedo[i]  # Activated values, with autograd.
```

`ide_degree` determines the input dimension automatically: degrees 1 through 5
produce 4, 10, 20, 38 and 72 features. IDE uses a fixed harmonic basis; there are
no trainable SH coefficients stored per voxel. The normalized recurrence avoids
cancellation in expanded high-degree Legendre polynomials. Constants are module
buffers and move to the voxel device along with the MLP.

The normal is the negative density gradient at the voxel center. Only visible
rows of the gathered geometry are read. Below `normal_eps`, the normal is
undefined: shading uses a detached viewing-direction fallback with zero normal
gradients into geometry. Each voxel still has one color per view, rather than
one independent color for every ray sample.

The color expression is `sigmoid(albedo) + sigmoid(tint) * specular`, with a
sigmoid at the MLP output. It is learned directly against loaded image values;
no linear/sRGB conversion, tone mapping or RGB padding is applied. Values can
exceed one. `color_unclamped` is the image at the target resolution used for
photometric training, while `color` is clipped for display and evaluation.
`raw_color` retains the supersampled resolution for existing consumers.

Training uses a voxel optimizer for density and all four appearance tensors,
plus an Adam optimizer for the shared network. The latter and its scheduler
survive voxel pruning/subdivision. The learning rates are `geo_lr`, `albedo_lr`,
`roughness_lr`, `tint_lr`, `bottleneck_lr` and `specular_lr` in `src/config.py`.
All four raw appearance tensors follow the voxel masks and subdivision order;
children inherit their parent's values.

## Checkpoints and compatibility

Version 2 checkpoints save the raw appearance tensors, shared network weights,
architecture and geometry. Optional quantization compresses geometry and voxel
appearance, while retaining network weights in full precision. Loading rebuilds
the saved architecture, so render scripts need not guess the IDE/MLP dimensions.
`--save_optimizer` also stores both optimizers and both schedulers, with their
iteration. `--load_optimizer` requires the matching model iteration. Resuming
continues at the next iteration and retains the learning-rate schedule.

Old configuration keys are migrated when reading YAML: `sh0_lr` becomes
`albedo_lr`, `sh0_init` becomes `albedo_init`, and `reset_sh_ckpt` becomes
`reset_albedo_ckpt`. SH degree and higher-order initialization options are removed.

Loading an old SH checkpoint emits a warning: its constant SH color initializes
albedo, while directional coefficients are discarded. Specular appearance is
newly initialized and needs training. This is an approximate warm start, not an
exact reproduction of the original SH images. Legacy optimizer moments cannot
be resumed in the new architecture.

Mesh coloring uses the `albedo` mode. The viewer offers `ref` and `albedo` instead
of an SH-degree slider. `render_by_trace_sdf.py` is a legacy entry point whose
SDF tracing backend is absent from this repository; it now reports that missing
capability explicitly. Use `render.py` for the supported volume renderer.

## Validation

```bash
python -m unittest discover -s tests -v
```

The suite uses CPU reference octree/gather operations to test the actual Python
model, shader, optimizers and persistence without requiring CUDA rasterization.
It compares IDE against SciPy, checks gradients, all layout initialization paths,
visible/empty subsets, adaptation, quantization, migration and unclipped training
images. The independent reference uses SciPy 1.15 or newer. A separate GPU test
executes the real rasterizer, backward pass, optimizer steps and checkpoint
roundtrip when CUDA is available. CPU checks alone do not establish rendering
quality, GPU memory use or speed; compare those against the SH baseline on held
out views after training.

## Memory controls

`model.color_chunk_size` defaults to 32768 visible voxels per shading block.
`model.color_checkpoint` defaults to true: training recomputes IDE/MLP
activations during backward rather than retaining all blocks' intermediate
tensors. This trades computation for memory without changing the color model
or stopping its gradients. Evaluation under `torch.no_grad()` still uses blocks
but does not need checkpoint recomputation. These runtime controls are not
restored from checkpoint metadata, so smaller blocks can be selected when
loading a model on another GPU.

For a conservative starting point on 12 GB, append `cfg/rtx3060_12gb.yaml` after
the dataset YAML. It uses degree 3 (20 IDE features), disables supersampling and
the initial subdivision of all eligible voxels, and caps the total at one
million voxels. These choices reduce model capacity and antialiasing; their
quality/performance tradeoff must be measured on the target scene. The profile
is not a measured guarantee of fitting every scene in 12 GB.

If memory is still insufficient, set `--color_chunk_size 16384` and/or
`--subdivide_max_num 500000`. A smaller block reduces shading workspaces;
the voxel cap also reduces persistent model/optimizer and rasterizer storage.
