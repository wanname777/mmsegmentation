_base_ = [
    '../_base_/datasets/my_acdc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MyUnet',
        in_channels=3,
        # out_channels=64,
        # num_stages=5,
        # strides=(1, 1, 1, 1, 1),
        # enc_num_convs=(2, 2, 2, 2, 2),
        # dec_num_convs=(2, 2, 2, 2),
        # downsamples=(True, True, True, True),
        # enc_dilations=(1, 1, 1, 1, 1),
        # dec_dilations=(1, 1, 1, 1),
        # with_cp=False,
        # conv_cfg=None,
        # norm_cfg=norm_cfg,
        # act_cfg=dict(type='ReLU'),
        # upsample_cfg=dict(type='InterpConv'),
        # norm_eval=False),
    ),
    decode_head=dict(
        type='MyFCNHead',
        in_channels=64,
        in_index=-1,
        channels=64,
        # num_convs=1,
        # concat_input=False,
        # dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0,
                 avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0,
                 avg_non_ignore=True)]

        # align_corners=False,
        # loss_decode=dict(
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(128,128), stride=(85, 85)))

# model = dict(backbone=dict(
#                 type='MyUnet',
#                 in_channels=3,
#                 base_channels=64,
#             ),
#             test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
evaluation = dict(metric='mDice')
