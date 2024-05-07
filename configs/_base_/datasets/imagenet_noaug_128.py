def get_classses(directory_path):
    import os

    # Get a list of all items in the directory
    all_items = os.listdir(directory_path)

    # Filter to include only directories (folders), excluding files
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

    # print("Folder Names:", folder_names)

    return folder_names




# dataset settings
dataset_type = 'mmcls.ImageNet'
#dataset_type = 'mmcls.CustomDataset'
# different from mmcls, we adopt the setting used in BigGAN.
# Importantly, the `to_rgb` is set to `False` to remain image orders as BGR.
# Remove `RandomFlip` augmentation and change `RandomCropLongEdge` to
# `CenterCropLongEdge` to elminiate randomness.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=2,
    train=dict(
        classes = get_classses('data/imagenet/train'),
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = get_classses('data/imagenet/val'),
        data_prefix='data/imagenet/val',
        # ann_file='data/imagenet/val/val_annotations.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        classes = get_classses('data/imagenet/val'),
        data_prefix='data/imagenet/val',
        # ann_file='data/imagenet/val/val_annotations.txt',
        pipeline=test_pipeline)
        )
