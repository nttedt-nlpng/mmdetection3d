_base_ = [
    'mmdet3d::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.OccFormer.occformer'])
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 2]

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [2, 2, 2]

class_names = ['empty', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
num_class = len(class_names)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_z = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0],
               point_cloud_range[3],
               voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1],
               point_cloud_range[4],
               voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2],
               point_cloud_range[5],
               voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

# settings for mask2former head
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3  # divided by ndim
mask2former_num_heads = voxel_out_channels // 32

default_scope = 'mmdet3d'
model = dict(
    type='OccupancyFormer',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        # original DCNv2 will print log when perform load_state_dict
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]
    ),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False
    ),
    img_bev_encoder_backbone=dict(
        type='OccupancyEncoder',
        num_stage=len(voxel_num_layer),
        in_channels=numC_Trans,
        block_numbers=voxel_num_layer,
        block_inplanes=voxel_channels,
        block_strides=voxel_strides,
        out_indices=voxel_out_indices,
        with_cp=True,
        norm_cfg=norm_cfg
    ),
    img_bev_encoder_neck=dict(
        type='MSDeformAttnPixelDecoder3D',
        strides=[2, 4, 8, 16],
        in_channels=voxel_channels,
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        norm_cfg=norm_cfg,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=voxel_out_channels,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None
                ),
                ffn_cfgs=dict(
                    embed_dims=voxel_out_channels
                ),
                feedforward_channels=voxel_out_channels * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')
            ),
            init_cfg=None
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=voxel_out_channels // 3,
            normalize=True
        ),
    ),
    pts_bbox_head=dict(
        type='Mask2FormerHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        num_queries=mask2former_num_queries,
        num_things_classes=num_class,
        num_stuff_classes=0,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        # using stand-alone pixel decoder
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=mask2former_pos_channel,
            normalize=True
        ),
        # using the original transformer decoder
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False
                ),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True
                ),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')
            ),
            init_cfg=None
        )
    )
)

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes/'
nus_class_meta = 'configs/nuscenes.yaml'

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

# simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)
backend_args = None

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5)

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles',
         is_train=True,
         data_config=data_config,
         img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', dataset='nusc'),
    dict(type='LoadNuscOccupancyAnnotations',
         is_train=True,
         grid_size=occ_size,
         point_cloud_range=point_cloud_range,
         bda_aug_conf=bda_aug_conf,
         cls_metas=nus_class_meta),
    dict(type='Pack3DDetInputs',
         keys=('img_inputs', 'gt_occ', 'points_occ'),
         meta_keys=('pc_range', 'occ_size'))
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles',
         is_train=False,
         data_config=data_config,
         img_norm_cfg=img_norm_cfg),
    dict(type='LoadNuscOccupancyAnnotations',
         is_train=False,
         grid_size=occ_size,
         point_cloud_range=point_cloud_range,
         bda_aug_conf=bda_aug_conf,
         cls_metas=nus_class_meta),
    dict(type='Pack3DDetInputs',
         keys=('img_inputs', 'gt_occ', 'points_occ'),
         meta_keys=('pc_range', 'occ_size', 'sample_idx', 'timestamp',
                    'scene_token', 'img_filenames', 'scene_name'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        occ_size=occ_size,
        pc_range=point_cloud_range)
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        occ_size=occ_size,
        pc_range=point_cloud_range)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        occ_size=occ_size,
        pc_range=point_cloud_range)
)

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# TODO evaluation
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        },
        norm_decay_mult=0.0
    )
)

# learning rate policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[20, 23]
)

# hook
default_hooks = dict(
    logger=dict(type='LoggerHook',
                interval=50,
                backend_args=backend_args),
    checkpoint=dict(type='CheckpointHook',
                    max_keep_ckpts=1,
                    interval=1)
)
