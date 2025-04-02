import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import random

class Dataset_preprocess_train:
    #初始化了缩放图像base大小，裁剪补全大小，水平翻转概率，以及标准化的均值&方差参数
    def __init__(self,base_size,crop_size,hflip_prob=0.5,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        
        #需要进行随机裁剪，翻转，缩放，张量化，标准化
        self.trans=[
            RandomResize(min_size , max_size),
            RandomCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean,std=std)
        ]
        if hflip_prob>0:
            self.trans.append(RandomHorizonFlip(hflip_prob))

        #实例化 Compose类 对象
        self.transforms=Compose(self.trans)

    def __call__(self, img , mask):
        #调用
        return self.transforms(img,mask)
    

class Dataset_preprocess_test:
    def __init__(self,base_size,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        #需要进行张量化，标准化
        self.trans=[
            RandomResize(base_size, base_size),
            ToTensor(),
            Normalize(mean=mean,std=std) # 这句应该是不起作用的
        ]
        self.transforms=Compose(self.trans)

    def __call__(self, img, mask):
        return self.transforms(img,mask)

#对输入的image和mask进行数据处理
def get_transforms(train:bool=True):
    base_size=520
    crop_size=480

    #对训练集进行数据处理
    if train:
        return Dataset_preprocess_train(base_size,crop_size) # 其他保持default即可
    else:
        return Dataset_preprocess_test(base_size)
    
#crop时调用，以免出现边长小的情况
def pad_fill(img,size,fill:int=0):
    if min(img.size)<size:
        ow,oh=img.size
        padw=size-ow if ow<size else 0
        padh=size-oh if oh<size else 0
        img=F.pad(img,(0,0,padw,padh),fill=fill)

    return img
    

class Compose:
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img, mask):
        #transforms是各种预处理类对象的call调用
        for t in self.transforms:
            img,mask=t(img,mask)
        return img,mask


class RandomResize:
    def __init__(self,min_size,max_size):
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self, img, mask):
        size=random.randint(self.min_size,self.max_size)
        img=F.resize(img,size) # resize方法默认是双线性插值，这对于原图像是合适的
        #对于标签，由于固定像素值有对应的类别，所以采用最邻近插值
        mask=F.resize(mask,size,interpolation=transforms.InterpolationMode.NEAREST)

        return img,mask


class RandomHorizonFlip:
    def __init__(self,hflip_prod):
        self.hflip_prod=hflip_prod

    def __call__(self, img,mask):
        if random.random()<self.hflip_prod:
            img,mask=F.hflip(img),F.hflip(mask)
        return img,mask

#需要对边长不到base_size的进行
class RandomCrop:
    def __init__(self,crop_size):
        self.crop_size=crop_size
        

    def __call__(self, img, mask):
        if min(img.size)<self.crop_size:
            img=pad_fill(img,self.crop_size,0) # 原始图像默认填充0
        if min(mask.size)<self.crop_size:
            mask=pad_fill(mask,self.crop_size,255) # mask图像填充的部分不能参与损失函数的计算，所以填充255（后续计算时忽略）
        crop_params=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
        img=F.crop(img,*crop_params)
        mask=F.crop(mask,*crop_params)

        return img,mask

class ToTensor:
    def __call__(self, img,mask):
        img=F.to_tensor(img) # 会将img像素值缩放到0-1之间
        mask=torch.as_tensor(np.array(mask),dtype=torch.int64)
        return img,mask


class Normalize:
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    
    def __call__(self, img,mask):
        img=F.normalize(img,mean=self.mean,std=self.std)
        # mask 像素值固定 0-21 & 255 不用标准化
        return img,mask

