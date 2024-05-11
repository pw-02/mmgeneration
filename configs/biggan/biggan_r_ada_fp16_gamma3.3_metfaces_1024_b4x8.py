_base_ = [
    '../_base_/models/biggan/biggan_128x128.py',
    '../_base_/datasets/ffhq_flip.py', '../_base_/default_runtime.py'
]


imgs_root = 'data/metfaces/images/'
data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root=imgs_root)),
    val=dict(imgs_root=imgs_root))

# # define dataset
# # you must set `samples_per_gpu`
# data = dict(samples_per_gpu=4, workers_per_gpu=1)

# adjust running config
lr_config = None
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=20)
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

total_iters = 125000 #500000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

# Note set your inception_pkl's path
inception_pkl = 'work_dir/imagenet.pkl'
evaluation = dict(
    type='GenerativeEvalHook',
    interval=5000, #10000
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            inception_pkl=inception_pkl,
            bgr2rgb=True),
        dict(type='IS', num_images=50000)
    ],
    sample_kwargs=dict(sample_model='ema'),
    best_metric=['fid', 'is'])

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        bgr2rgb=True),
    is50k=dict(type='IS', num_images=50000))



# _base_ = [
#     '../_base_/models/stylegan/stylegan3_base.py',
#     '../_base_/datasets/ffhq_flip.py', '../_base_/default_runtime.py'
# ]

# synthesis_cfg = {
#     'type': 'SynthesisNetwork',
#     'channel_base': 65536,
#     'channel_max': 1024,
#     'magnitude_ema_beta': 0.999,
#     'conv_kernel': 1,
#     'use_radial_filters': True
# }
# r1_gamma = 3.3  # set by user
# d_reg_interval = 16

# load_from = 'https://download.openmmlab.com/mmgen/stylegan3/stylegan3_r_ffhq_1024_b4x8_cvt_official_rgb_20220329_234933-ac0500a1.pth'  # noqa

# # ada settings
# aug_kwargs = {
#     'xflip': 1,
#     'rotate90': 1,
#     'xint': 1,
#     'scale': 1,
#     'rotate': 1,
#     'aniso': 1,
#     'xfrac': 1,
#     'brightness': 1,
#     'contrast': 1,
#     'lumaflip': 1,
#     'hue': 1,
#     'saturation': 1
# }

# model = dict(
#     type='StaticUnconditionalGAN',
#     generator=dict(
#         out_size=1024,
#         img_channels=3,
#         rgb2bgr=True,
#         synthesis_cfg=synthesis_cfg),
#     discriminator=dict(
#         type='ADAStyleGAN2Discriminator',
#         in_size=1024,
#         input_bgr2rgb=True,
#         data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs, ada_kimg=100)),
#     gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
#     disc_auxiliary_loss=dict(loss_weight=r1_gamma / 2.0 * d_reg_interval))

# imgs_root = 'data/metfaces/images/'
# data = dict(
#     samples_per_gpu=4,
#     train=dict(dataset=dict(imgs_root=imgs_root)),
#     val=dict(imgs_root=imgs_root))

# ema_half_life = 10.  # G_smoothing_kimg

# ema_kimg = 10
# ema_nimg = ema_kimg * 1000
# ema_beta = 0.5**(32 / max(ema_nimg, 1e-8))

# custom_hooks = [
#     dict(
#         type='VisualizeUnconditionalSamples',
#         output_dir='training_samples',
#         interval=5000),
#     dict(
#         type='ExponentialMovingAverageHook',
#         module_keys=('generator_ema', ),
#         interp_mode='lerp',
#         interp_cfg=dict(momentum=ema_beta),
#         interval=1,
#         start_iter=0,
#         priority='VERY_HIGH')
# ]

# inception_pkl = 'work_dirs/inception_pkl/metface_1024x1024_noflip.pkl'
# metrics = dict(
#     fid50k=dict(
#         type='FID',
#         num_images=50000,
#         inception_pkl=inception_pkl,
#         inception_args=dict(type='StyleGAN'),
#         bgr2rgb=True))

# evaluation = dict(
#     type='GenerativeEvalHook',
#     interval=dict(milestones=[100000], interval=[10000, 5000]),
#     metrics=dict(
#         type='FID',
#         num_images=50000,
#         inception_pkl=inception_pkl,
#         inception_args=dict(type='StyleGAN'),
#         bgr2rgb=True),
#     sample_kwargs=dict(sample_model='ema'))

# lr_config = None

# total_iters = 160000
