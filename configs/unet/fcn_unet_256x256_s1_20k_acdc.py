_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/my_acdc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    test_cfg=dict(mode='whole', crop_size=(128,128), stride=(85, 85)))
evaluation = dict(metric='mDice')
