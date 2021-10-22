# -*- coding: utf-8 -*-
# @Time : 2021/9/1 10:27 
# @Author : jiangwei hao 
# @File : dataloader.py 
# @Software: PyCharm
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io,transform
import numpy as np
import torch
from torchvision import transforms, utils
from PIL import Image
# 重写数据加载
def read_label(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    newlines = [line.strip().split(" ", 1) for line in lines]
    # strip()返回移除字符串头尾指定的字符生成的新字符串。(防止有空格)
    # print('newlines:', len(newlines))
    return newlines

class KeyPointDataset(Dataset):
    def __init__(self, label_file, root_dir, transform=None):
        self.keypoint = read_label(label_file)
        # print(self.keypoint)
        self.root_dir = root_dir
        # print(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.keypoint)

    def __getitem__(self, idx):
        # __getitem__做的事情就是返回第index个样本的具体数据:
        # print('idx:',idx)
        img_name = os.path.join(self.root_dir, self.keypoint[idx][0])
        # print(img_name)
        image = io.imread(img_name) # 尺寸为 H*W*C,
        # print(image.shape)

        kp = self.keypoint[idx][1]
        kpnum = [int(p) for p in kp.split(" ")]
        kp = kpnum[:12]

        cls = kpnum[12:]
        cls = np.array(cls)
        # cls = cls.astype('float')

        # cls = torch.from_numpy(cls)
        kp = np.array(kp)
        # print('kp:', kp)
        kp = kp.astype("float").reshape(-1, 2)
        # print('kp2:', kp)
        sample = image, kp
        if self.transform:
            sample = self.transform(sample)
        # print('kp3:',sample[1])
        return sample, cls


# 实现简单的预处理转换
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, kp = sample[0], sample[1]

        h, w = image.shape[:2]
        # print(h,w)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        #         print("orign shape w: {} h: {} {}".format(w,h,kp))

        kp = kp * [self.output_size[1] / w, self.output_size[0] / h]
        #         print("after w: {} h: {} {}".format(self.output_size[1],self.output_size[0],kp))
        # print('1:',image[0][0])
        img = transform.resize(image, (new_h, new_w)) # 数值的取值范围是（0~1）自动归一化
        # print(new_h, new_w)
        # print('2:', img[0][0])

        return img, kp

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), kp


class RandomBrightness(object):
    def __init__(self, delta=2):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, kp

class RandomContrast(object):
    def __init__(self, lower=0.95, upper=1.05):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, kp


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, kp

class RandomHue(object):
    def __init__(self, delta=0.001):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, kp

class RandomSaturation(object):
    def __init__(self, lower=0.95, upper=1.05):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, kp


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        numpy数组到tensor的变化，另外还有维度的变化。
    """

    def __call__(self, sample):
        image, kp = sample[0], sample[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(kp)
        return image.type(torch.FloatTensor), label.type(torch.FloatTensor)


if __name__ == '__main__':
    train_txt_path = "/workspace1/data/keypoint/0915_keypoint_4/train_cls_4.txt"
    train_images = "/workspace1/data/keypoint/0915_keypoint_4/train/"
    val_txt_path = "/workspace1/data/keypoint/0910_keypoint_two/val.txt"
    val_images = "/workspace1/data/keypoint/0910_keypoint_two/val/"
    kptrain = KeyPointDataset(label_file=train_txt_path,root_dir=train_images,transform=transforms.Compose([Rescale((128,128)),ToTensor()]))
    # platekpval = KeyPointDataset(label_file="./PlateKeyPoint/valmul.txt",root_dir="./PlateKeyPoint/val",transform=transforms.Compose([Rescale((128,256)),ToTensor()]))



    trainloader = DataLoader(kptrain, batch_size=1, shuffle=True, num_workers=1)
    print(len(trainloader))
    print(type(trainloader))
    print('-------')


    for i, (sample, plate) in enumerate(trainloader):
        images, labels = sample[0], sample[1]
        print(images.shape)
        print(images[0][0][0])
        print('P:',plate)
        print('plate:',type(plate))
        print('labels:',type(labels))

        print('num imgs:',len(images))
        print('labels:', labels[0])
        print('num:',len(labels[0]))
        print('shape:', labels.shape)
        print('i:',i)
        print('----------------------------------------------')



    # dataset = [1, 2, 3, 4, 5]
    # a = iter(dataset)
    # for i in range(5):
    #     data = next(a)
    #     print(data)