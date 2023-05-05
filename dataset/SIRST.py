import cv2
import os

import numpy as np
from torch.utils.data.dataset import Dataset

from config import config


class G1G2Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.imageset_dir = os.path.join('<path to SIRST>/training_img/')
            self.imageset_gt_dir = os.path.join('<path to SIRST>/training_mask/')
        elif self.mode == 'test.py':
            self.imageset_dir = os.path.join('<path to SIRST>/test_img/')
            self.imageset_gt_dir = os.path.join('<path to SIRST>/test_mask/')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 341
        elif self.mode == 'test':
            return 86
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_dir = os.path.join(self.imageset_dir, "%d.png" % (idx+1))
            gt_dir = os.path.join(self.imageset_gt_dir, "%d.png" % (idx+1))

            # 
            real_input = np.float32(cv2.imread(img_dir, 1)) / 255.0
            real_input = cv2.resize(real_input, (224, 224))

            if config.ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:, :, 2] * 2 - 1
                input_images = np.expand_dims(input_images, axis=0)

            bufImg = cv2.imread(gt_dir, -1)
            bufImg = cv2.resize(bufImg, (224, 224))
            dilated_bufImg = bufImg
            output_images = np.float32(dilated_bufImg) / 255.0  
            output_images = np.expand_dims(output_images, axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            return sample_info

        elif self.mode == 'test.py':
            img_dir = os.path.join(self.imageset_dir, "%03d.png" % idx)
            gt_dir = os.path.join(self.imageset_gt_dir, "%03d.png" % idx)

            # 
            real_input = np.float32(cv2.imread(img_dir, 1)) / 255.0
            real_input = cv2.resize(real_input, (224, 224))

            if config.ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:, :, 2] * 2 - 1
                input_images = np.expand_dims(input_images, axis=0)

            bufImg = cv2.imread(gt_dir, -1)
            bufImg = cv2.resize(bufImg, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
            dilated_bufImg = bufImg
            output_images = np.float32(dilated_bufImg) / 255.0 
            output_images = np.expand_dims(output_images, axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            return sample_info
        else:
            raise NotImplementedError
            
            
            
