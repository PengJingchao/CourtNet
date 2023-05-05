import cv2
import os

import numpy as np
from torch.utils.data.dataset import Dataset

from config import config


class G1G2Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.imageset_dir = os.path.join('<path to MFIRST>/training/')
            self.imageset_gt_dir = os.path.join('<path to MFIRST>/training/')
        elif self.mode == 'test.py':
            self.imageset_dir = os.path.join('<path to MFIRST>/test_org/')
            self.imageset_gt_dir = os.path.join('<path to MFIRST>/test_gt/')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 9900
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_dir = os.path.join(self.imageset_dir, "%06d_1.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%06d_2.png" % idx)

            #
            real_input = np.float32(cv2.imread(img_dir, -1)) / 255.0
            real_input = cv2.resize(real_input, (224, 224))

            if config.ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:, :, 2] * 2 - 1
                input_images = np.expand_dims(input_images, axis=0)

            bufImg = cv2.imread(gt_dir, -1)
            bufImg = cv2.resize(bufImg, (224, 224))
            dilated_bufImg = bufImg
            output_images = np.float32(dilated_bufImg) / 255.0  # 像素归一化
            output_images = np.expand_dims(output_images, axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            return sample_info

        elif self.mode == 'test':
            img_dir = os.path.join(self.imageset_dir, "%05d.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png" % idx)

            #
            real_input = np.float32(cv2.imread(img_dir, -1)) / 255.0
            real_input = cv2.resize(real_input, (224, 224))

            if config.ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:, :, 2] * 2 - 1
                input_images = np.expand_dims(input_images, axis=0)

            bufImg = cv2.imread(gt_dir, -1)
            bufImg = cv2.resize(bufImg, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
            dilated_bufImg = bufImg
            output_images = np.float32(dilated_bufImg) / 255.0  # 
            output_images = np.expand_dims(output_images, axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            return sample_info
        else:
            raise NotImplementedError
            
            
            
