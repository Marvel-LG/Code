import math
import random
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import cv2
from data.util.synthetic import online_add_degradation_v2
from data.util.photoshop import Photoshop
from data.util.mask import get_irregular_mask
import torch.nn.functional as F

import struct

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def zero_mask(size):
    x = np.zeros((size, size, 3)).astype('uint8')
    mask = Image.fromarray(x).convert("RGB")
    return mask


class BigFileMemoryLoader(object):
    def __init__(self, file_path, image_size=None):
        super(BigFileMemoryLoader, self).__init__()
        self.file_path = file_path
        self.img_names = sorted(os.listdir(file_path))
        self.img_num = len(self.img_names)
        self.image_size = image_size

    def __getitem__(self, index):
        index = index % self.img_num
        try:
            img_path = os.path.join(self.file_path, self.img_names[index])
            img = Image.open(img_path).convert('RGB')

            if self.image_size is not None:
                img = self.random_crop(img, size=self.image_size)
            return self.img_names[index], img
        except Exception as e:
            print(f'Image read error for index {index}: {self.img_names[index]} - {e}')
            return self.__getitem__((index + 1) % self.img_num)

    def getitem_from_name(self, img_name):
        if img_name in self.img_names:
            mask = Image.open(os.path.join(self.file_path, img_name))
            mask = np.array(mask)
            # mask = np.where(mask > 125, 255, 0).astype(np.uint8)
            mask = Image.fromarray(mask).convert('RGB')
        else:
            mask = None
        return mask

    def random_crop(self, img, size):
        width, height = img.size
        new_width, new_height = size
        if width < new_width or height < new_height:
            raise ValueError("Image size is smaller than the crop size.")

        left = np.random.randint(0, width - new_width + 1)
        top = np.random.randint(0, height - new_height + 1)
        img = img.crop((left, top, left + new_width, top + new_height))
        return img

    def __len__(self):
        return self.img_num


class PairBigFileMemoryLoader(object):
    def __init__(self, data_root, random=False):
        super(PairBigFileMemoryLoader, self).__init__()
        self.dir_test = data_root
        self.image_path = os.path.join(self.dir_test, 'images')
        self.mask_path = os.path.join(self.dir_test, 'masks')

        self.loaded_test_images = BigFileMemoryLoader(self.image_path)
        self.mask_paths = sorted(os.listdir(self.mask_path))

        self.img_num = len(self.loaded_test_images)
        self.random = random

    def __getitem__(self, index):
        try:
            img_name, image = self.loaded_test_images[index]
            if img_name in self.mask_paths:
                mask = Image.open(os.path.join(self.mask_path, img_name))
                mask = np.array(mask)
                # mask = np.where(mask > 125, 255, 0).astype(np.uint8)
                mask = Image.fromarray(mask).convert('RGB')
                assert (image.size == mask.size), "image and mask size mismatch"
            else:
                mask = None
                # if self.random:
                #     if len(self.mask_paths) and random.random() < 0.5:
                #         mask = Image.open(os.path.join(self.mask_path, self.mask_paths[
                #             random.randint(0, len(self.mask_paths) - 1)])).convert('RGB')
                #         angle = random.uniform(0, 360)
                #         mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)
                #         mask = mask.resize(image.size, Image.LANCZOS)
                #         mask_array = np.array(mask)
                #         binary_mask = np.where(mask_array > 125, 255, 0).astype(np.uint8)
                #         mask = Image.fromarray(binary_mask).convert('RGB')
                #     else:
                #         mask = get_irregular_mask(image.size)
                # else:
                #     mask = Image.new('RGB', image.size, (0, 0, 0))
                #     # print(
                #     #     f'Real_RGB_Old {os.path.join(self.mask_path, img_name)} haven\'t pair mask, return total zero mask')
            return image, mask, img_name
        except Exception:
            print('Image read error for index %d: %s' % (index, img_name))
            return self.__getitem__((index + 1) % self.img_num)

    def __len__(self):
        return self.img_num


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], pattern=None, phase='train', batch_size=1):
        self.phase = phase
        self.using_synthetic = (self.phase == 'test' and True)  # 测试阶段也使用合成数据
        self.dir_AB = data_root
        self.load_clear_VOC_pair = os.path.join(self.dir_AB, "Clear_VOC2012_Pair")
        self.load_clear_DIV2K_pair = os.path.join(self.dir_AB, "Clear_DIV2K_Pair")
        self.load_real_L_old_pair = os.path.join(self.dir_AB, "Real_L_Pair")
        self.load_real_RGB_old_pair = os.path.join(self.dir_AB, "Real_RGB_Pair")  # 测试的照片位置
        self.load_eval_pair = os.path.join(self.dir_AB, "Eval_Pair")
        self.load_test_pair = os.path.join(self.dir_AB, "Test_Old_Pair")
        self.load_mask = os.path.join(self.dir_AB, "Mask")
        if self.phase == 'train':
            self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_clear_VOC_pair)
            # self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_real_RGB_old_pair)
            # self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_clear_DIV2K_pair, random=True)
            # self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_eval_pair, random=False)
        else:
            self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_clear_VOC_pair)
            # self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_test_pair)
            # self.loaded_clear_Main_pair = PairBigFileMemoryLoader(self.load_real_RGB_old_pair)

        self.loaded_real_L_old_Pair = PairBigFileMemoryLoader(self.load_real_L_old_pair)
        self.loaded_real_RGB_old_pair = PairBigFileMemoryLoader(self.load_real_RGB_old_pair)
        self.loaded_mask = BigFileMemoryLoader(self.load_mask, image_size=image_size)

        self.image_size = image_size

        self.tfs = transforms.Compose([  # Transform处理组
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_tfs = transforms.Compose([
            # transforms.RandomCrop(image_size),  # 添加随机裁剪
            transforms.ToTensor(),  # 将图像转换为张量
        ])
        # self.ps = transforms.Compose(
        #     [
        #         Photoshop(spot_flag=True, scratch_flag=True, flwa_flag=True, phase=phase)
        #     ]
        # )
        self.image_size = image_size  # 图片大小
        ###################
        self.pattern = pattern
        self.or_syn_old_y = 0
        self.batch_size = batch_size
        self.batch_index = 0
        self.mask = None
        self.max_length_limit = None

    def get_synthetic(self, groundtruth):
        return online_add_degradation_v2(groundtruth)

    def __getitem__(self, index):
        if self.phase == 'train' or self.using_synthetic:
            degradation = None
            # 确定退化和采样数据集
            if self.or_syn_old_y >= 2 and (self.pattern == 'A' or self.pattern == 'AandMAP'):  # 干净图
                sampled_dataset = self.loaded_clear_Main_pair
                is_real_old_y = 2
            elif self.or_syn_old_y == 0 and (self.pattern == 'A' or self.pattern == 'AandMAP'):  # 取真实图像
                if random.uniform(0, 1) < 0.1:
                    sampled_dataset = self.loaded_real_L_old_Pair  # 从 L 中采样
                else:
                    sampled_dataset = self.loaded_real_RGB_old_pair  # 从 RGB 中采样
                is_real_old_y = 0
            else:  # 取得合成图
                sampled_dataset = self.loaded_clear_Main_pair  # 合成图
                degradation = 1
                is_real_old_y = 1
            if self.batch_index % self.batch_size == 0:
                if len(self.loaded_mask) and random.random() < 0:
                    mask_name, self.mask = self.loaded_mask[random.randint(0, len(self.loaded_mask) - 1)]
                else:
                    self.mask = get_irregular_mask(self.image_size)
            self.batch_index = (self.batch_index + 1) % self.batch_size  # 切换mask
            self.or_syn_old_y = (self.or_syn_old_y + 1) % 2  # 切换状态
            # 随机选择图像
            if index > len(sampled_dataset) - 1:
                sampled_dataset_len = len(sampled_dataset)
                index = random.randint(0, sampled_dataset_len - 1)
            if degradation is not None:
                B, mask, img_name = sampled_dataset[index]  # 获取图像名和图像数据
                A = online_add_degradation_v2(B, phase=self.phase)
            else:
                A, mask, img_name = sampled_dataset[index]
                B = A
            if mask is None:
                mask = self.mask

            if random.uniform(0, 1) < 0.1:
                A = A.convert("L")
                B = B.convert("L")
                A = A.convert("RGB")
                B = B.convert("RGB")
            # 裁剪图像
            A, B, mask = self.random_crop_images(A, B, mask, self.image_size)
            # 处理掩码

            mask_abs = np.abs(np.array(A) - np.array(B)).clip(0, 200).astype(np.uint8)

            if is_real_old_y == 0:
                mask_mean = np.full_like(mask, np.array(mask).clip(0, 200).mean())
                mask = np.maximum(mask_mean, self.mask)
            else:
                mask_mean = np.full_like(mask_abs, mask_abs.mean())
                mask = np.maximum(mask_mean, np.array(mask))
            # 转换为张量
            A_tensor = self.tfs(A)  # 合成图像
            B_tensor = self.tfs(B)  # 原图像
            mask_tensor = self.mask_tfs(Image.fromarray(mask))
        else:
            # 加载图像和掩膜

            # img, mask, img_name = self.loaded_real_RGB_old_pair[index]
            # img, mask, img_name = self.loaded_real_L_old_Pair[index]
            img, mask, img_name = self.loaded_clear_Main_pair[index]
            if mask is None:
                mask = Image.new('RGB', img.size, (0, 0, 0))
            img, img, mask = self.random_crop_images(img, img, mask, self.image_size)
            # 获取图像的宽度和高度
            width, height = img.size

            # 设置需要被整除的数字
            divisor = 8  # 可以根据需要更改

            # 计算新的宽度和高度，使其能够被 divisor 整除
            new_width = (width // divisor) * divisor
            new_height = (height // divisor) * divisor

            # 确保新的宽度和高度至少为 divisor
            if new_width == 0:
                new_width = divisor
            if new_height == 0:
                new_height = divisor

            # 调整图像和掩膜的大小
            img = img.resize((new_width, new_height), Image.LANCZOS)
            mask = mask.resize((new_width, new_height), Image.LANCZOS)

            # 将图像和掩膜转换为张量
            A_tensor = B_tensor = self.tfs(img)
            mask_tensor = self.mask_tfs(mask)

            # 设置标志
            is_real_old_y = 1
        mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
        ret = {}
        ret['gt_image'] = B_tensor
        ret['synthetic'] = A_tensor
        ret['mask'] = mask_tensor  # torch.where(mask_tensor == 1.0, torch.tensor(1.0), torch.tensor(0.0))
        # mask_tensor#torch.where(mask_tensor >= 0.95, torch.tensor(1.0), torch.tensor(0.0))
        ret['inst'] = is_real_old_y
        ret['path'] = img_name
        ret['mask_abs'] = torch.where(mask_tensor == 1.0, torch.tensor(1.0), torch.tensor(0.0))
        return ret

    def __len__(self):
        if self.max_length_limit is not None:
            return self.max_length_limit
        else:
            return len(self.loaded_clear_Main_pair)

    def random_crop_images(self, A, B, mask, size):
        # 检查 A 的大小
        if A.size[0] < size[0] or A.size[1] < size[1]:
            # 计算新的尺寸
            new_size = (max(A.size[0], size[0]), max(A.size[1], size[1]))
            # 放大 A、B 和 mask 到 new_size
            A = A.resize(new_size, Image.LANCZOS)
            B = B.resize(new_size, Image.LANCZOS)
            mask = mask.resize(new_size, Image.LANCZOS)

        # 随机裁剪的位置
        i = random.randint(0, A.size[1] - size[1])
        j = random.randint(0, A.size[0] - size[0])

        # 裁剪区域
        crop_box = (j, i, j + size[0], i + size[1])

        # 裁剪图像
        A_cropped = A.crop(crop_box)
        B_cropped = B.crop(crop_box)
        if mask.size != tuple(size):
            mask_cropped = mask.crop(crop_box)
        else:
            mask_cropped = mask

        return A_cropped, B_cropped, mask_cropped


class RealDatasets(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], pattern=None, phase='train', batch_size=None):
        self.phase = phase
        self.dir_AB = 'datasets/Test_Old_Pair/'  # data_root
        self.image_size = image_size
        self.load_degradation = os.path.join(self.dir_AB, "input")
        self.load_mask = os.path.join(self.dir_AB, "mask")
        self.load_step1 = os.path.join(self.dir_AB, "step1")
        self.load_step2 = os.path.join(self.dir_AB, "step2")

        self.loaded_degradation = BigFileMemoryLoader(self.load_degradation)
        self.loaded_mask = BigFileMemoryLoader(self.load_mask)
        self.loaded_step1 = BigFileMemoryLoader(self.load_step1)
        self.loaded_step2 = BigFileMemoryLoader(self.load_step2)

        self.tfs = transforms.Compose([  # Transform处理组
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_tfs = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
        ])
        ###################
        self.pattern = pattern
        self.or_syn_old_y = 0
        self.batch_size = batch_size
        self.batch_index = 0
        self.mask = None
        self.max_length_limit = None

    def __getitem__(self, index):
        is_real_old_y = 0
        if self.pattern == 'MAP':
            if self.phase == 'test':
                degradation_name, A = self.loaded_degradation[index]
            else:
                degradation_name, A = self.loaded_step1[index]
        else:
            # degradation_name, A = self.loaded_degradation[index]
            degradation_name, A = self.loaded_step1[index]
        mask = self.loaded_mask.getitem_from_name(degradation_name)
        B = self.loaded_step2.getitem_from_name(degradation_name)#GT

        if self.phase == 'train':
            A, B, mask = self.random_crop_images(A, B, mask, self.image_size)
            # A, B, mask = self.resize_images(A, B, mask, self.image_size)
        else:
            A, B, mask = self.resize_images(A, B, mask, [384, 384])
        mask_abs = np.abs(np.array(A) - np.array(B)).clip(0, 200).astype(np.uint8)
        mask_mean = np.full_like(mask_abs, mask_abs.mean())
        mask = np.maximum(mask_mean, np.array(mask))
        A_tensor = self.tfs(A)  # 合成图像
        B_tensor = self.tfs(B)  # 原图像
        mask_tensor = self.mask_tfs(Image.fromarray(mask))

        mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
        ret = {}
        ret['gt_image'] = B_tensor
        ret['synthetic'] = A_tensor
        ret['mask'] = mask_tensor  # torch.where(mask_tensor == 1.0, torch.tensor(1.0), torch.tensor(0.0))
        # mask_tensor#torch.where(mask_tensor >= 0.95, torch.tensor(1.0), torch.tensor(0.0))
        ret['inst'] = is_real_old_y
        ret['path'] = degradation_name
        ret['mask_abs'] = torch.where(mask_tensor == 1.0, torch.tensor(1.0), torch.tensor(0.0))
        return ret

    def __len__(self):
        if self.max_length_limit is not None:
            return self.max_length_limit
        else:
            return len(self.loaded_degradation)

    def random_crop_images(self, A, B, mask, size):
        # 检查 A 的大小
        if A.size[0] < size[0] or A.size[1] < size[1]:
            # 计算新的尺寸
            new_size = (max(A.size[0], size[0]), max(A.size[1], size[1]))
            # 放大 A、B 和 mask 到 new_size
            A = A.resize(new_size, Image.LANCZOS)
            B = B.resize(new_size, Image.LANCZOS)
            mask = mask.resize(new_size, Image.LANCZOS)

        # 随机裁剪的位置
        i = random.randint(0, A.size[1] - size[1])
        j = random.randint(0, A.size[0] - size[0])

        # 裁剪区域
        crop_box = (j, i, j + size[0], i + size[1])

        # 裁剪图像
        A_cropped = A.crop(crop_box)
        B_cropped = B.crop(crop_box)
        if mask.size != tuple(size):
            mask_cropped = mask.crop(crop_box)
        else:
            mask_cropped = mask

        return A_cropped, B_cropped, mask_cropped

    def resize_images(self, A, B, mask, size):
        # 假设 A, B, mask 都是形状为 (b, c, h, w) 的四维张量

        # 调整图像大小
        A_resized = A.resize(size,Image.BILINEAR)
        B_resized = B.resize(size,Image.BILINEAR)
        mask_resized = mask.resize(size,Image.NEAREST)

        return A_resized, B_resized, mask_resized


if __name__ == '__main__':
    # dataset = InpaintDataset('datasets', image_size=[256, 256], pattern='A', phase='train')
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #
    #     print(data)
    dataset = RealDatasets('/media/lwg/系统/Software/ComfyUI/output/', image_size=[256, 256], pattern='A',
                           phase='train')
    for i in range(len(dataset)):
        data = dataset[i]

        print(data)
