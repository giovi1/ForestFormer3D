_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']

# TreeScanPL10K fine-tuning setup.
#
# The model head is intentionally kept at the ForAINetV2-compatible 3-class
# shape so the existing checkpoint can be loaded directly. TreeScan training
# labels currently use only class 0 = background and class 1 = tree.

data_root_forainetv2 = 'data/TreeScanPL10K/'
work_dir = 'work_dirs/treescan_finetune_voxel_0_1_decoder'
load_from = 'work_dirs/clean_forestformer/epoch_3000_fix.pth'

# Decoder fine-tuning is much more memory-hungry than the previous preparation
# stage. These values keep the first TreeScan decoder run feasible on a 40 GB
# GPU while preserving the 0.1 m converted input resolution.
fine_tune_radius = 12
fine_tune_num_points = 320000
fine_tune_queries = 128

model = dict(
    voxel_size=0.1,
    radius=fine_tune_radius,
    query_point_num=fine_tune_queries,
    query_selection=dict(
        mode='isa',
        num_queries=fine_tune_queries,
        isa_ratio=0.34,
        multiscale_ratio=0.33,
        density_ratio=0.33,
        density_sampling_mode='balanced',
        density_deterministic=True,
        density_k=16,
        density_eps=1e-6,
        deduplicate_queries=True,
        fallback_to_isa=True,
        log_interval=100),
    chunk=1500,
    # The base ForAINetV2 config waits 1000 epochs before training the
    # instance decoder. TreeScan fine-tuning is shorter, so enable it from
    # the first epoch.
    prepare_epoch=-1,
    # Start with a recall-friendly threshold; use cfg-options to ablate this.
    score_th=0.2,
    test_cfg=dict(
        output_dir=work_dir,
        step_size=6,
        sp_score_thr=0.05,
        npoint_thr=5,
        min_instance_points=5,
        edge_filter=False,
        suppress_background_instances=False,
        debug_full_scene=True))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=fine_tune_radius),
    dict(type='GridSample', grid_size=0.2),
    dict(type='PointSample_', num_points=fine_tune_num_points),
    dict(type='SkipEmptyScene_'),
    dict(type='PointInstClassMapping_', num_classes=3),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask', 'ratio_inspoint', 'vote_label',
            'instance_mask'
        ])
]

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=fine_tune_radius),
    dict(type='GridSample', grid_size=0.2),
    dict(type='PointSample_', num_points=fine_tune_num_points),
    dict(type='PointInstClassMapping_', num_classes=3),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask', 'instance_mask'
        ])
]

full_scene_eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask', 'instance_mask'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    prefetch_factor=1,
    dataset=dict(
        data_root=data_root_forainetv2,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=2,
    dataset=dict(
        data_root=data_root_forainetv2,
        pipeline=full_scene_eval_pipeline))

test_dataloader = dict(
    num_workers=2,
    dataset=dict(
        data_root=data_root_forainetv2,
        pipeline=full_scene_eval_pipeline))

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='mRQ',
        rule='greater',
        save_optimizer=True),
    logger=dict(type='LoggerHook', interval=20),
    visualization=dict(type='Det3DVisualizationHook', draw=False))
