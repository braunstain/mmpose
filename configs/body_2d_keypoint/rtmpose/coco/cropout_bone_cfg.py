# project_baseline_safe.py

_base_ = ['../../../_base_/default_runtime.py']

# ===== Runtime =====
max_epochs = 20
base_lr = 5e-4  # gentle fine-tune
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# ===== Optimizer =====
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        # keep BN/bias out of weight decay
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True
    )
)

# ===== LR schedule (simple + stable) =====
# short warmup then cosine
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# IMPORTANT: disable auto scaling so LR is what you think it is
auto_scale_lr = dict(enable=False)

# ===== Codec =====
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False
)

# ===== Model =====
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        frozen_stages=4,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        # this init_cfg is overwritten by load_from (fine to leave)
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'
        )
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False
        ),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True
        ),
        struct_loss=dict(
            type='BoneVectorSimCCLoss',
            loss_weight=0.01
        ),
        decoder=codec
    ),
    # keep this consistent with your baseline eval
    test_cfg=dict(flip_test=True)
)


# ===== Datasets =====
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/train2017/'
data_root_och = 'data/OCHuman/'
backend_args = dict(backend='local')

# ===== Pipelines =====
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.0,
        scale_factor=[0.9, 1.1],
        rotate_factor=20
    ),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]
    ),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_och,
        data_mode=data_mode,
        ann_file='annotations/val_split/ochuman_non_zero.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# ===== Hooks =====
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=20)
)

# EMA helps stabilize short fine-tunes
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.001,
        update_buffers=True,
        priority=49
    )
]

# ===== Evaluator =====
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_och + 'annotations/val_split/ochuman_non_zero.json'
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_och + 'annotations/test_split/ochuman_non_zero.json'
)

# ===== Start from official RTMPose-M pose weights =====
load_from = 'checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
resume = False