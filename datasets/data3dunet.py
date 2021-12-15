import os
import cv2
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from resources.config import get_configs

config = get_configs()

path = config.dataset_path


def rotate(image, angle, scale=1.0):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (height, width))
    return rotated


class MyPatientFor3DUNet(Dataset):
    def __init__(self, in_channels=1, out_channels=1, step='train', data_type='float32'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_type = data_type
        self.data_len = 0
        self.mask_list = []
        self.step = step
        self.folder_path_mask = ""
        self.folder_path_PET = ""
        if step == 'train':
            self.folder_path_mask = path + "/train/mask/"
            self.folder_path_PET = path + "/train/image/"
            self.folder_path_mask_ = path + "/train/mask_/"
            self.mask_list = os.listdir(self.folder_path_mask)
            self.mask_list = natsorted(self.mask_list)
            self.data_len = len(self.mask_list)
        elif step == 'val':
            self.folder_path_mask = path + "/val/mask1/"
            self.folder_path_mask_ = path + "/val/mask_1/"
            self.folder_path_PET = path + "/val/image1/"
            self.mask_list = os.listdir(self.folder_path_mask)
            self.mask_list = natsorted(self.mask_list)
            self.data_len = len(self.mask_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        raw = cv2.imread(self.folder_path_PET + self.mask_list[idx], 0) / 255
        mask = cv2.imread(self.folder_path_mask + self.mask_list[idx], 0)
        mask_ = cv2.imread(self.folder_path_mask_ +
                           self.mask_list[idx], 0) / 255
        # mask_ = mask

        raw = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_ = cv2.resize(mask_, (256, 256), interpolation=cv2.INTER_NEAREST)

        # raw = raw[4:-4,4:-4]
        # mask = mask[4:-4,4:-4]
        raw = (raw - raw.mean()) / (raw.std() + 0.00000001)

        # angle = random.randint(-45, 45)
        #
        # aug1_raw = rotate(raw, angle)
        # aug1_mask = rotate(mask, angle)
        # aug2_raw = cv2.flip(raw, 1)
        # aug2_mask = cv2.flip(mask, 1)
        #
        # aug1_raw = np.array(aug1_raw)[np.newaxis, :, :]
        # aug1_mask = np.array(aug1_mask)[np.newaxis, :, :]
        # aug2_raw = np.array(aug2_raw)[np.newaxis, :, :]
        # aug2_mask = np.array(aug2_mask)[np.newaxis, :, :]
        #
        # aug1_raw = torch.Tensor(aug1_raw)
        # aug1_mask = torch.Tensor(aug1_mask)
        # aug2_raw = torch.Tensor(aug2_raw)
        # aug2_mask = torch.Tensor(aug2_mask)
        #
        [X, Y] = mask.shape
        raw = torch.Tensor(raw)
        mask = torch.Tensor(mask)
        mask_ = torch.Tensor(mask_)
        self.raw = torch.zeros((self.in_channels, X, Y))
        self.mask = torch.zeros((self.in_channels, X, Y))
        self.mask_ = torch.zeros((self.in_channels, X, Y))
        self.raw[0] = raw
        self.mask[0] = mask
        self.mask_[0] = mask_

        # if self.step == 'train':
        #     return self.raw, self.mask, aug1_raw, aug1_mask, aug2_raw, aug2_mask
        # if self.step == 'val':
        #     return self.raw, self.mask

        return self.raw, self.mask, self.mask_
