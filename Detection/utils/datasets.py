from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor
import glob
import concurrent.futures
import multiprocessing
import random
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import torch

from torchvision.transforms import v2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

import time

class RGBTFolders(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.thermalfiles = sorted(glob.glob('%s/thermal/*.*' % folder_path))
        self.rgbfiles = sorted(glob.glob('%s/rgb/*.*' % folder_path))
        self.labelfiles = sorted(glob.glob('%s/labels/*.*' % folder_path))
        self.img_shape = (img_size, img_size)
        self.max_objects = 10

    def __getitem__(self, index):
        imgrgb_path = self.rgbfiles[index % len(self.rgbfiles)]
        imgthermal_path = self.thermalfiles[index % len(self.thermalfiles)]
        imglabel_path = self.labelfiles[index % len(self.thermalfiles)]
        # Extract image
        imgrgb = np.array(Image.open(imgrgb_path))
        imgthermal = np.array(Image.open(imgthermal_path))
        imgthermal = self.fill_thermal(imgthermal)
        
        imgthermal = np.expand_dims(imgthermal, axis=-1)
        imglabel = np.array(Image.open(imglabel_path))
        imglabel = np.expand_dims(imglabel, axis=-1)
        h, w, _ = imgrgb.shape # same shapes of images rgb, label and thermal
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_imgrgb = np.pad(imgrgb, pad, 'constant', constant_values=127.5) / 255.
        input_imgthermal = np.pad(imgthermal, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_imgrgb.shape
        # input_imglabel = np.pad(imglabel, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_imgrgb = resize(input_imgrgb, (*self.img_shape, 3), mode='reflect')
        input_imgthermal = resize(input_imgthermal, (*self.img_shape, 1), mode='reflect')
        # input_imglabel = resize(input_imglabel, (*self.img_shape, 1), mode='reflect')
        # Concatenate  images
        input_img = np.concatenate((input_imgrgb, input_imgthermal), axis=-1)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        # Find contours in the segmentation image
        contours, _ = cv.findContours(imglabel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Iterate through the contours to get bounding boxes
        labels = np.zeros((self.max_objects, 5))
        labels[:, 0] = 255 # setting undetected bbox as [255, 0, 0, 0, 0]
        num_of_deleted = 0
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            if w < 20 and h < 10:
                num_of_deleted += 1
                continue # skipping small bboxes, cause the model won't learn it anyway
            x_center = (x + x + w) / (2 * padded_w)
            y_center = (y + y + h) / (2 * padded_h)
            x_center += pad[1][0] / padded_w
            y_center += pad[0][0] / padded_h
            bbox_width = w / padded_w
            bbox_height = h / padded_h
            unique_pixels, counts = np.unique(imglabel[y:y+h, x:x+w], return_counts=True)
            max = 0
            j_max = -1
            for j, pix in enumerate(unique_pixels):
                if pix == 0 or pix == 255:
                    continue
                if counts[j] > max:
                    max = counts[j]
                    j_max = j
            object_class = 256
            if j_max != -1:
                object_class = unique_pixels[j_max]
            if i < self.max_objects:
                labels[i, :] = [object_class - 1, x_center, y_center, bbox_width, bbox_height]

        #Deletes the path to an image, that doesn't meet criteria
        if len(contours) == num_of_deleted:
            del self.rgbfiles[index % len(self.rgbfiles)]
            del self.thermalfiles[index % len(self.thermalfiles)]
            del self.labelfiles[index % len(self.thermalfiles)]
            return self.__getitem__(index)
        else:
            return imgrgb_path, input_img, labels

    def __len__(self):
        return len(self.thermalfiles)
    
    def fill_thermal(self, thermal_image):
        hole_mask = (thermal_image == 0).astype(np.uint8)
        filled_thermal = cv.inpaint(
            thermal_image, 
            hole_mask, 
            10, 
            cv.INPAINT_TELEA
        )
        return filled_thermal

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


class RGBTFolderz(Dataset):

    def __init__(self, folder_path, img_size=416, train=False):
        self.thermalfiles = sorted(glob.glob('%s/thermal/*.*' % folder_path))
        self.rgbfiles = sorted(glob.glob('%s/rgb/*.*' % folder_path))
        self.labelfiles = sorted(glob.glob('%s/labels/*.*' % folder_path))
        self.img_shape = (img_size, img_size)
        self.max_objects = 10
        self.label = np.zeros((len(self.thermalfiles),self.max_objects,5))
        self.input_imgs = np.zeros(shape=(len(self.thermalfiles),4,img_size,img_size))
        self.train = train
        
        mark_for_deletion = []
        for ii, (RGB_T, RGB, LABEL) in enumerate(zip(tqdm(self.thermalfiles),self.rgbfiles,self.labelfiles)):
            imgrgb = Image.open(RGB)
            imgthermal = Image.open(RGB_T)
            imglabel = Image.open(LABEL)

            if self.train:
                totensor = v2.ToTensor()
                imgconcat = torch.concatenate((totensor(imgrgb), totensor(imgthermal), totensor(imglabel)), 0)
                transforms = v2.Compose([
                    v2.RandomResizedCrop(size=(720, 1280), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation(degrees=(-30, 30)),
                ])
                imgconcat = transforms(imgconcat)
                topilimage = v2.ToPILImage()
                imgrgb = np.array(topilimage(imgconcat[0:3,:,:]))
                imgthermal = np.array(topilimage(imgconcat[3,:,:]))
                imglabel = np.array(topilimage(imgconcat[4,:,:]))
            else:
                imgrgb = np.array(imgrgb)
                imgthermal = np.array(imgthermal)
                imglabel = np.array(imglabel)
            imgthermal = self.fill_thermal(imgthermal)
            imgthermal = np.expand_dims(imgthermal, axis=-1)
            imglabel = np.expand_dims(imglabel, axis=-1)

            h, w, _ = imgrgb.shape # same shapes of images rgb, label and thermal
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

            # Add padding
            input_imgrgb = np.pad(imgrgb, pad, 'constant', constant_values=127.5) / 255.
            input_imgthermal = np.pad(imgthermal, pad, 'constant', constant_values=127.5) / 255.
            padded_h, padded_w, _ = input_imgrgb.shape
            # Resize and normalize
            input_imgrgb = resize(input_imgrgb, (*self.img_shape, 3), mode='reflect')
            input_imgthermal = resize(input_imgthermal, (*self.img_shape, 1), mode='reflect')
            # Concatenate  images
            input_img = np.concatenate((input_imgrgb, input_imgthermal), axis=-1)
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))

            self.input_imgs[ii] = input_img

            # Find contours in the segmentation image
            contours, _ = cv.findContours(imglabel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # Iterate through the contours to get bounding boxes
            labels = np.zeros((self.max_objects, 5))
            # labels[:, 0] = 255 # setting undetected bbox as [255, 0, 0, 0, 0]
            num_of_deleted = 0
            for i, contour in enumerate(contours):
                x, y, w, h = cv.boundingRect(contour)
                if w < 20 and h < 10:
                    num_of_deleted += 1
                    continue # skipping small bboxes, cause the model won't learn it anyway
                x_center = (x + x + w) / (2 * padded_w)
                y_center = (y + y + h) / (2 * padded_h)
                x_center += pad[1][0] / padded_w
                y_center += pad[0][0] / padded_h
                bbox_width = w / padded_w
                bbox_height = h / padded_h
                unique_pixels, counts = np.unique(imglabel[y:y+h, x:x+w], return_counts=True)
                max = 0
                j_max = -1
                for j, pix in enumerate(unique_pixels):
                    if pix == 0 or pix == 255:
                        continue
                    # if pix not in {2, 3, 4}:
                    #     print("here")
                    if counts[j] > max:
                        max = counts[j]
                        j_max = j
                object_class = 256
                if j_max != -1:
                    object_class = unique_pixels[j_max]
                if i < self.max_objects:
                    labels[i - num_of_deleted, :] = [object_class - 1, x_center, y_center, bbox_width, bbox_height]
            self.label[ii] = labels

            if len(contours) == num_of_deleted:
                mark_for_deletion.append(ii)
        
        for mark in tqdm(mark_for_deletion[::-1]):
            del self.thermalfiles[mark]
            del self.rgbfiles[mark]
            del self.labelfiles[mark]
            self.label = np.delete(self.label,mark,axis=0)
            self.input_imgs = np.delete(self.input_imgs,mark,axis=0)

    def __getitem__(self, index):
        # As pytorch tensor
        input_img = torch.from_numpy(self.input_imgs[index]).float()

        return self.rgbfiles[index % len(self.rgbfiles)], input_img, self.label[index]

    def __len__(self):
        return len(self.thermalfiles)
    
    def fill_thermal(self, thermal_image):
        hole_mask = (thermal_image == 0).astype(np.uint8)
        filled_thermal = cv.inpaint(
            thermal_image, 
            hole_mask, 
            10, 
            cv.INPAINT_TELEA
        )
        return filled_thermal


class RGBTFolderX(Dataset):
    def __init__(self, folder_path, img_size=416, train=False):
        self.thermalfiles = sorted(glob.glob('%s/thermal/*.*' % folder_path))
        self.rgbfiles = sorted(glob.glob('%s/rgb/*.*' % folder_path))
        self.labelfiles = sorted(glob.glob('%s/labels/*.*' % folder_path))
        self.img_shape = (img_size, img_size)
        self.max_objects = 10

    def __getitem__(self, index):
        img_rgb_path = self.rgbfiles[index % len(self.rgbfiles)]
        img_thermal_path = self.thermalfiles[index % len(self.thermalfiles)]
        img_label_path = self.labelfiles[index % len(self.thermalfiles)]

        img_rgb = np.array(Image.open(img_rgb_path))/255
        img_thermal = np.array(Image.open(img_thermal_path))/255
        img_thermal = np.expand_dims(img_thermal, axis=-1)
        img_label = np.loadtxt(img_label_path)
        input_img = np.concatenate((img_rgb, img_thermal), axis=-1)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        #As PyTorch Tensor
        input_img = torch.from_numpy(input_img).float()

        return img_rgb_path, input_img, img_label

    def __len__(self):
        return len(self.thermalfiles)
    


class RGBTFolderP(Dataset):

    def __init__(self,folder_path, img_size=416, train=False):
        self.thermalfiles = sorted(glob.glob('%s/thermal/*.*' % folder_path))
        self.rgbfiles = sorted(glob.glob('%s/rgb/*.*' % folder_path))
        self.labelfiles = sorted(glob.glob('%s/labels/*.*' % folder_path))
        self.img_shape = (img_size, img_size)
        self.max_objects = 10
        self.label = np.zeros((len(self.thermalfiles),self.max_objects,5))
        self.input_imgs = np.zeros(shape=(len(self.thermalfiles),4,img_size,img_size))
        self.train = train
        self.num_of_deleted = 0
        self.mark_for_deletion = np.zeros(shape=(len(self.thermalfiles),))
        self._preprocess_images()


    def __getitem__(self,index):
        #As PyTorch Tensor
        input_img = torch.from_numpy(self.input_imgs[index]).float()
        return self.rgbfiles[index % len(self.rgbfiles)], input_img, self.label[index]


    def __len__(self):
        return len(self.thermalfiles)


    def _load_and_process_image(self, ii):
        imgrgb = Image.open(self.rgbfiles[ii])
        imgthermal = Image.open(self.thermalfiles[ii])
        imglabel = Image.open(self.labelfiles[ii])

        if self.train:
            totensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            imgconcat = torch.concatenate((totensor(imgrgb), totensor(imgthermal), totensor(imglabel)), 0)
            transforms = v2.Compose([
                v2.RandomResizedCrop(size=(720, 1280), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-30, 30)),
            ])
            imgconcat = transforms(imgconcat)
            topilimage = v2.ToPILImage()
            imgrgb = np.array(topilimage(imgconcat[0:3,:,:]))
            imgthermal = np.array(topilimage(imgconcat[3,:,:]))
            imglabel = np.array(topilimage(imgconcat[4,:,:]))
        else:
            imgrgb = np.array(imgrgb)
            imgthermal = np.array(imgthermal)
            imglabel = np.array(imglabel)
        imgthermal = self.fill_thermal(imgthermal)
        imgthermal = np.expand_dims(imgthermal, axis=-1)
        imglabel = np.expand_dims(imglabel, axis=-1)

        h, w, _ = imgrgb.shape # same shapes of images rgb, label and thermal
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        
        # Add padding
        input_imgrgb = np.pad(imgrgb, pad, 'constant', constant_values=127.5) / 255.
        input_imgthermal = np.pad(imgthermal, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_imgrgb.shape
        # input_imglabel = np.pad(imglabel, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_imgrgb = resize(input_imgrgb, (*self.img_shape, 3), mode='reflect')
        input_imgthermal = resize(input_imgthermal, (*self.img_shape, 1), mode='reflect')
        # input_imglabel = resize(input_imglabel, (*self.img_shape, 1), mode='reflect')
        # Concatenate  images
        input_img = np.concatenate((input_imgrgb, input_imgthermal), axis=-1)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # Find contours in the segmentation image
        contours, _ = cv.findContours(imglabel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Iterate through the contours to get bounding boxes
        labels = np.zeros((self.max_objects, 5))
        # labels[:, 0] = 255 # setting undetected bbox as [255, 0, 0, 0, 0]
        num_of_deleted = 0
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            if w < 20 and h < 10:
                num_of_deleted += 1
                continue # skipping small bboxes, cause the model won't learn it anyway
            x_center = (x + x + w) / (2 * padded_w)
            y_center = (y + y + h) / (2 * padded_h)
            x_center += pad[1][0] / padded_w
            y_center += pad[0][0] / padded_h
            bbox_width = w / padded_w
            bbox_height = h / padded_h
            unique_pixels, counts = np.unique(imglabel[y:y+h, x:x+w], return_counts=True)
            max = 0
            j_max = -1
            for j, pix in enumerate(unique_pixels):
                if pix == 0 or pix == 255:
                    continue
                if counts[j] > max:
                    max = counts[j]
                    j_max = j
            object_class = 256
            if j_max != -1:
                object_class = unique_pixels[j_max]
            if i < self.max_objects:
                labels[i - num_of_deleted, :] = [object_class - 1, x_center, y_center, bbox_width, bbox_height]
        
        delete = False
        if len(contours) == num_of_deleted:
            delete = True

        return delete, input_img, labels, self.rgbfiles[ii], self.thermalfiles[ii], self.labelfiles[ii]
    

    def _how_many_for_deletion(self, result_of_pi):
        for delete,_,_,_,_,_  in result_of_pi:
            if delete:
                self.num_of_deleted += 1
    

    def _clear_records(self):
        new_length = len(self) - self.num_of_deleted
        self.thermalfiles = [None]*new_length
        self.rgbfiles = [None]*new_length
        self.labelfiles = [None]*new_length
        self.label = np.zeros(shape=(new_length,self.max_objects,5))
        self.input_imgs = np.zeros(shape=(new_length,4,self.img_shape[0],self.img_shape[1]))


    def _delete_record(self, results):
        i = 0
        for dels,imgs,labl,rgbf,therf,labelf in tqdm(results):
            if not dels:
                self.thermalfiles[i] = therf
                self.rgbfiles[i] = rgbf
                self.labelfiles[i] = labelf
                self.label[i] = labl
                self.input_imgs[i] = imgs
                i += 1


    def _preprocess_images(self):
        task = [i for i in range(len(self.rgbfiles))]

        with ThreadPoolExecutor() as executor:
            # Execute tasks in parallel with ThreadPoolExecutor
            results = list(tqdm(executor.map(self._load_and_process_image, task), total=len(task)))

        self._how_many_for_deletion(results)
        self._clear_records()
        self._delete_record(results)
        

    def fill_thermal(self, thermal_image):
        hole_mask = (thermal_image == 0).astype(np.uint8)
        filled_thermal = cv.inpaint(
            thermal_image, 
            hole_mask, 
            10, 
            cv.INPAINT_TELEA
        )
        return filled_thermal
    