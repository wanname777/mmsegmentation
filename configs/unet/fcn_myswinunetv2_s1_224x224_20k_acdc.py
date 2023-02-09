_base_ = [
    '../_base_/datasets/my_acdc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
img_size=(224,224)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MySwinUnetV2',
        # load_pretrain_path='./checkpoint/swin_tiny_patch4_window7_224.pth',
        # drop_path_rate=0.1,ape=True
        # img_size=img_size, embedding_dim=96, window_size=7,
        # dropout=0.2,depth=2
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=-1,
        channels=96,
        num_classes=4,
        kernel_size=1,
        num_convs=1,
        norm_cfg=norm_cfg,
        # ignore_index=0,
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, avg_non_ignore=True)]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(128, 128), stride=(85, 85)))
evaluation = dict(metric='mDice')
