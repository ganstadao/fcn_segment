from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from utils.transforms import *


class VOCdataset(Dataset):
    def __init__(self,root_path,train:bool=True,transforms=None):
        super(VOCdataset,self).__init__()
        mask_path=os.path.join(root_path,'SegmentationClass')
        image_path=os.path.join(root_path,'JPEGImages')

        #image中的图片数量远大于mask
        mask_name=os.listdir(mask_path)
        #筛选出匹配的项
        image_name=[i.replace('.png','.jpg') for i in mask_name]
        self.mask_list=[os.path.join(mask_path,i) for i in mask_name]
        self.image_list=[os.path.join(image_path,i) for i in image_name]
        self.transforms=transforms

        for i in self.image_list:
            if not os.path.exists(i):
                print(f"file {i} not exists!")

    def __getitem__(self, index):
        img_path=self.image_list[index]
        mask_path=self.mask_list[index]

        img=Image.open(img_path) # mode=RGB
        mask=Image.open(mask_path) # mode=P

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img,mask
        

    def __len__(self):
        return len(self.mask_list)
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
    
def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
    


'''root_dir=".\data\VOCdevkit\VOC2012"
test_dataset=VOCdataset(root_dir,True,get_transforms(True))
print(len(test_dataset))'''

