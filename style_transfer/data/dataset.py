import sys
import torch.utils.data as data
from os import listdir
from anything_in_anyscene.style_transfer.utils.tools import default_loader, is_image_file, normalize
import os
import torch
import torchvision.transforms as transforms
import cv2
import random
import numpy as np
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, return_name=False):
        super(Dataset, self).__init__()
        # if with_subfolder:
        #     self.samples = self._find_samples_in_subfolders(data_path)
        # else:
        #     self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        # self.random_crop = random_crop
        self.return_name = return_name
        self.ground_truth_samples = [x for x in listdir(os.path.join(data_path,'raw')) if is_image_file(x)]
        self.scene_img_samples = [x for x in listdir(os.path.join(data_path,'scene')) if is_image_file(x)]
        self.mask_img_samples = [x for x in listdir(os.path.join(data_path,'mask')) if is_image_file(x)]
        self.obj_segment_samples = [x for x in listdir(os.path.join(data_path,'obj_segments')) if is_image_file(x)]
        intersection = set(self.ground_truth_samples) & set(self.scene_img_samples) & set(self.mask_img_samples) & set(self.obj_segment_samples)
        self.samples = list(intersection)

    def __getitem__(self, index):
        # path = os.path.join(self.data_path, self.samples[index])
        ground_truth_path = os.path.join(self.data_path, 'raw', self.samples[index])
        scene_img_path = os.path.join(self.data_path, 'scene', self.samples[index])
        mask_img_path = os.path.join(self.data_path, 'mask', self.samples[index])
        obj_segment_path = os.path.join(self.data_path, 'obj_segments', self.samples[index])

        ground_truth = default_loader(ground_truth_path)

        scene_img_arr = np.array(default_loader(scene_img_path))

        mask_img_ori = default_loader(mask_img_path)
        dilate_radius = random.randint(2,10)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        dilate_mask_arr = cv2.dilate(np.array(mask_img_ori), dilate_kernel)
        mask_img = Image.fromarray(dilate_mask_arr)
        
        scene_img_arr[dilate_mask_arr==255] = 0
        scene_img = Image.fromarray(scene_img_arr)

        erode_radius = random.randint(1,5)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
        erode_mask = cv2.erode(np.array(mask_img_ori), erode_kernel)

        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        obj_segment = color_jitter(default_loader(obj_segment_path))
        obj_segment_arr = np.array(obj_segment)
        # obj_segment_arr = np.array(default_loader(obj_segment_path))
        obj_segment_arr[erode_mask==0] = 0
        obj_segment = Image.fromarray(obj_segment_arr)
      
        ground_truth = transforms.ToTensor()(ground_truth)
        scene_img = transforms.ToTensor()(scene_img)
        mask_img = transforms.ToTensor()(mask_img)
        mask_img = mask_img[:1,:,:] #对于三通道的mask图片只取单通道
        obj_segment = transforms.ToTensor()(obj_segment)
        
        
        # 使用索引将 B 中的对应元素替换到 A 中
        

        # color_jitter_scene_img = scene_img.clone()
        # nonzero_indices = torch.nonzero(mask_img)
        # color_jitter_scene_img[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] = \
        # obj_segment[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

        color_jitter_scene_img = scene_img.clone()
        nonzero_indices = torch.nonzero(mask_img)
        for index in nonzero_indices:
            y, x = index[1], index[2]
            obj_segment_value = obj_segment[:, y, x]
            # 替换 color_jitter_scene_img 中的对应元素
            color_jitter_scene_img[:, y, x] = obj_segment_value

        # if self.return_name:
        #     return self.samples[index], img
        # else:
        #     return img
        return ground_truth, scene_img, mask_img, obj_segment, color_jitter_scene_img
    
    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)
