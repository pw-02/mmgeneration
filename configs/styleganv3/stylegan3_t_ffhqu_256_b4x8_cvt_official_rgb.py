_base_ = ['./stylegan3_base.py']

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 16384,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        out_size=256,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(in_size=256, channel_multiplier=1))
