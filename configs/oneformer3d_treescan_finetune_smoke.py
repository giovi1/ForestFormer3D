_base_ = ['./oneformer3d_treescan_finetune.py']

work_dir = 'work_dirs/treescan_finetune_voxel_0_2_smoke'

model = dict(
    test_cfg=dict(output_dir=work_dir))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_optimizer=False),
    logger=dict(type='LoggerHook', interval=1))
