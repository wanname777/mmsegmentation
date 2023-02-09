from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyACDCDataset(CustomDataset):
    CLASSES = ('background', 'class1','class2','class3')
    # CLASSES = ('class1', 'class2', 'class3')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]
    # PALETTE = [[244, 35, 232], [70, 70, 70], [102, 102, 156]]
    def __init__(self, **kwargs):
        super(MyACDCDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_gt.png',
            **kwargs)
        assert self.file_client.exists(self.img_dir)

