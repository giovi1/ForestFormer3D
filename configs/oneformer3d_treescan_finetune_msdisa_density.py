_base_ = ['./oneformer3d_treescan_finetune.py']

work_dir = 'work_dirs/treescan_finetune_voxel_0_1_msdisa_density'

model = dict(
    query_selection=dict(
        mode='density',
        num_queries=128,
        isa_ratio=0.5,
        multiscale_ratio=0.0,
        density_ratio=0.5,
        density_sampling_mode='balanced',
        density_deterministic=True),
    test_cfg=dict(output_dir=work_dir))
