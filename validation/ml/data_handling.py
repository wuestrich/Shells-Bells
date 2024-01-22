#! /usr/bin/env python3.8
# coding: utf-8
# dataloader preparation adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

def load_bmp(d_frame, row:int, col:str, d_dir:str="."):
    """First loads a bmp in grayscale and then converts values to 0.1"""
    loaded_bmp = np.array(Image.open(f"{d_dir}/{d_frame.iloc[row][col]}").convert("L"))
    bmp = []
    for x in range(len(loaded_bmp)):
        row = []
        for y in range(len(loaded_bmp[x])):
            if loaded_bmp[x][y] == 30:
                row.append(0)
            else:
                row.append(1)
        bmp.append(row)
    return np.array(bmp)


def load_bmp_img(d_frame, row:int, col:str, d_dir:str="."):
    """Loads a bmp in grayscale"""
    return np.array(Image.open(f"{d_dir}/{d_frame.iloc[row][col]}").convert("L"))

def get_data_point(d_frame, i, data_dir):
    bmp_reference = load_bmp_img(d_frame,i, "reference", data_dir)
    bmp_constructed = load_bmp_img(d_frame, i, "constructed", data_dir)
    label = d_frame.iloc[i]["label"]
    label_class = d_frame.iloc[i]["label_class"]
    return (bmp_reference, bmp_constructed, label, label_class)

def show_bitmap(bitmap):
    """Plot a bitmap"""
    plt.imshow(bitmap.permute(1,2,0).numpy())

def show_bitmap_batch(sample_batched):
    """Show images of validation bitmaps for a batch of samples."""
    bitmap_batch, label_class_batch = sample_batched['bitmap'], sample_batched['label_class']
    batch_size = len(bitmap_batch)
    bmp_size = bitmap_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(bitmap_batch)
    #print(grid.shape)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("batch from dataloader")

class ValidationData(Dataset):

    def __init__(self, csv_file:str, im_dir:str, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            im_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info_frame = pd.read_csv(csv_file)
        self.root_dir = im_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_info_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #TODO: concatenation step is obsolete if generated data has this out of the box
        reference_bmp_name = f"{self.root_dir}/{self.data_info_frame.iloc[idx]['reference']}"
        constructed_bmp_name = f"{self.root_dir}/{self.data_info_frame.iloc[idx]['constructed']}"
        #reference_bmp = np.array(Image.open(reference_bmp_name).convert("L"))
        #constructed_bmp = np.array(Image.open(constructed_bmp_name).convert("L"))
        #validation_bmp = np.concatenate((reference_bmp, constructed_bmp), dtype="float32")
        reference_bmp = Image.open(reference_bmp_name).convert("L")
        constructed_bmp = Image.open(constructed_bmp_name).convert("L")
        validation_bmp = Image.new('L', (reference_bmp.width, reference_bmp.height + constructed_bmp.height))
        validation_bmp.paste(reference_bmp, (0, 0))
        validation_bmp.paste(constructed_bmp, (0, reference_bmp.height))
        validation_bmp = np.array(validation_bmp, dtype="float32")
        validation_bmp[validation_bmp==30] = 0
        validation_bmp[validation_bmp!=0] = 1
        validation_bmp = np.expand_dims(validation_bmp, axis=-1) # add dimension for grayscale image (channels=1) https://stackoverflow.com/questions/49237117/python-add-one-more-channel-to-image
        label = self.data_info_frame.iloc[idx]["label"]
        label_class = self.data_info_frame.iloc[idx]["label_class"]
        sample = {"bitmap":validation_bmp, "label": label, "label_class": label_class}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample and labels to Tensors."""
    def __call__(self, sample):
        bitmap, label, label_class_str = sample["bitmap"], sample["label"], sample["label_class"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        bitmap = bitmap.transpose((2,0,1))
        if label_class_str == "normal":
            label_class = [0]
        elif label_class_str == "spoofing":
            label_class = [1]
        elif label_class_str == "delay":
            label_class = [2]
        elif label_class_str == "early_execution":
            label_class = [3]
        else: # masquerading
            label_class = [4]
        # torch.tensor(bitmap).view(1, 256,646)
        #t = transforms.Grayscale()
        return {"bitmap": torch.from_numpy(bitmap), "label": torch.tensor(label), "label_class": torch.tensor(label_class)}

def no_batch():
    data_info_csv = "./ml_data/labels.csv"
    im_folder = "./ml_data"
    test_data_set = ValidationData(data_info_csv, im_folder, transforms.Compose([ToTensor()]))
    for i in range(len(test_data_set)):
        sample = test_data_set[i+10000]
        print(i+1000, sample["bitmap"].shape, sample["label"], sample["label_class"])
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(f"Sample {i+10000}: {sample['label_class']}")
        ax.axis("off")
        show_bitmap(sample["bitmap"])        
        if i == 3:
            plt.show()
            break

def batched():
    data_info_csv = "./ml_data/labels.csv"
    im_folder = "./ml_data"
    test_data_set = ValidationData(data_info_csv, im_folder, transforms.Compose([ToTensor()]))
    dataloader = DataLoader(test_data_set, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['bitmap'].size(),
              sample_batched['label_class'])
        #print(sample_batched)

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_bitmap_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

def main(args):
    no_batch()
    batched()
    pass


if __name__ == "__main__":
    main(sys.argv)