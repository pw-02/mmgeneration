model = dict(
    type='BasiccGAN',
    num_classes=10,
    generator=dict(
        type='BigGANGenerator',
        output_scale=32,
        noise_size=128,
        num_classes=10,
        base_channels=64,
        with_shared_embedding=False,
        sn_eps=1e-08,
        sn_style='torch',
        init_type='N02',
        split_noise=False,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=32,
        num_classes=10,
        base_channels=64,
        sn_eps=1e-08,
        sn_style='torch',
        init_type='N02',
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))
train_cfg = dict(
    disc_steps=4, gen_steps=1, batch_accumulation_steps=1, use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)))
dataset_type = 'mmcls.CIFAR10'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=25,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type='mmcls.CIFAR10',
            data_prefix='data/cifar10',
            pipeline=[
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='ToTensor', keys=['gt_label']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ])),
    val=dict(
        type='mmcls.CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='mmcls.CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=20)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=4,
        start_iter=4000,
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH')
]
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
lr_config = None
total_iters = 500000
use_ddp_wrapper = True
inception_pkl = None
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
        dict(type='IS', num_images=50000)
    ],
    sample_kwargs=dict(sample_model='ema'),
    best_metric=['fid', 'is'])
metrics = dict(
    fid50k=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    is50k=dict(type='IS', num_images=50000))
work_dir = 'logs'
gpu_ids = [0]
