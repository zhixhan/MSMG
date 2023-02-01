# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import random
import torch
 
class exchange_token:
    def __init__(self):
        pass

    def __call__(self, features, mask_box):
        b, c, h, w = features.size()
        assert h*w == 14*14
        new_idx, mask_x1, mask_x2, mask_y1, mask_y2 = mask_box
        features[:, :, mask_x1 : mask_x2, mask_y1 : mask_y2] = features[new_idx, :, mask_x1 : mask_x2, mask_y1 : mask_y2]
        return features
        
class get_mask_box:
    def __init__(self, shape='stripe', mask_size=1, mode='random_direct'):
        self.shape = shape
        self.mask_size = mask_size
        self.mode = mode

    def __call__(self, features):
        # Stripe mask
        if self.shape == 'stripe':
            if self.mode == 'horizontal':
                mask_box = self.hstripe(features, self.mask_size)
            elif self.mode == 'vertical':
                mask_box = self.vstripe(features, self.mask_size)
            elif self.mode == 'random_direction':
                if random.random() < 0.5:
                    mask_box = self.hstripe(features, self.mask_size)
                else:
                    mask_box = self.vstripe(features, self.mask_size)
            else:
                raise Exception("Unknown stripe mask mode name")
        # Square mask
        elif self.shape == 'square':
            if self.mode == 'random_size':
                self.mask_size = 4 if random.random() < 0.5 else 5
            mask_box = self.square(features, self.mask_size)
        # Random stripe/square mask
        elif self.shape == 'random':
            random_num = random.random()
            if random_num < 0.25:
                mask_box = self.hstripe(features, 2)
            elif random_num < 0.5 and random_num >= 0.25:
                mask_box = self.vstripe(features, 2)
            elif random_num < 0.75 and random_num >= 0.5:
                mask_box = self.square(features, 4)
            else:
                mask_box = self.square(features, 5)
        else:
            raise Exception("Unknown mask shape name")
        return mask_box

    def hstripe(self, features, mask_size):
        """
        """
        # horizontal stripe
        mask_x1 = 0
        mask_x2 = features.shape[2]
        y1_max = features.shape[3] - mask_size
        mask_y1 = torch.randint(y1_max, (1,))
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def vstripe(self, features, mask_size):
        """
        """
        # vertical stripe
        mask_y1 = 0
        mask_y2 = features.shape[3]
        x1_max = features.shape[2] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def square(self, features, mask_size):
        """
        """
        # square
        x1_max = features.shape[2] - mask_size
        y1_max = features.shape[3] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_y1 = torch.randint(y1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box