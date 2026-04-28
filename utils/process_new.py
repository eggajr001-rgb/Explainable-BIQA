import torch
import numpy as np


# === 1. 随机裁剪  ===
class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        d_img = sample['d_img_org']
        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size


        if h == new_h and w == new_w:
            ret_d_img = d_img
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            ret_d_img = d_img[:, top: top + new_h, left: left + new_w]

        # 使用 update 或直接赋值，不要创建新字典，以免丢失 d_label
        sample['d_img_org'] = ret_d_img
        return sample


# === 2. 归一化===
class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_img = (d_img - self.mean) / self.var
        sample['d_img_org'] = d_img
        return sample


# === 3. 随机水平翻转 ===
class RandHorizontalFlip(object):
    def __init__(self, prob_aug):
        self.prob_aug = prob_aug

    def __call__(self, sample):
        d_img = sample['d_img_org']
        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        # 随机决定是否翻转
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()

        sample['d_img_org'] = d_img
        return sample


# === 4. 转 Tensor (修复报错) ===
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']

        # 1. 处理图片
        sample['d_img_org'] = torch.from_numpy(d_img).type(torch.FloatTensor)

        # 2. [修复报错] 智能处理分数: 无论是 float 还是 numpy 都能转
        if isinstance(score, np.ndarray):
            sample['score'] = torch.from_numpy(score).type(torch.FloatTensor)
        else:
            sample['score'] = torch.tensor(score).type(torch.FloatTensor)

        # 3. [关键] 处理畸变标签
        if 'd_label' in sample:
            sample['d_label'] = torch.tensor(sample['d_label']).type(torch.LongTensor)

        return sample