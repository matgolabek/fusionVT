import argparse
import glob
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import shutil
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image
from tqdm import tqdm

def fill_thermal(thermal_image):
    hole_mask = (thermal_image == 0).astype(np.uint8)
    filled_thermal = cv.inpaint(
        thermal_image, 
        hole_mask, 
        10, 
        cv.INPAINT_TELEA
    )
    return filled_thermal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='./data/PST900_RGBT_Dataset', help='path to dataset')
    parser.add_argument('--image_size', type=int, default=416, help='TODO')
    parser.add_argument('--suffix', type=str, default='_MOD', help='TODO')
    opt = parser.parse_args()
    print(opt)

    folder_path = opt.image_folder
    folder_suffix = opt.suffix
    img_size = opt.image_size
    path_parts = Path(folder_path).parts
    path_to_mod_img = '.'+''.join('/'+name for name in path_parts) + folder_suffix
    print(f'New image directory : {path_to_mod_img}')

    if Path(path_to_mod_img).exists():
        x = ''
        while x != "y":
            x = input("Found a directory delete? [y/n]:")
            if x == 'n': return 0
        shutil.rmtree(path_to_mod_img)
        #return 0

    Path(path_to_mod_img).mkdir(parents=True)
    for name in ['/test', '/train']:
        print(path_to_mod_img+name)
        Path(path_to_mod_img+name).mkdir(parents=True)
        for subnames in ['/thermal','/rgb','/labels']:
            print(path_to_mod_img+name+subnames)
            Path(path_to_mod_img+name+subnames).mkdir(parents=True)
    
    for name in ['/train','/test']:
        dir_path = folder_path + name
        save_dir_path = path_to_mod_img + name
        thermalfiles = sorted(glob.glob('%s/thermal/*.*' % dir_path))
        rgbfiles = sorted(glob.glob('%s/rgb/*.*' % dir_path))
        labelfiles = sorted(glob.glob('%s/labels/*.*' % dir_path))
        img_shape = (img_size, img_size)
        max_objects = 10
        
        for ii, (RGB_T, RGB, LABEL) in enumerate(zip(tqdm(thermalfiles), rgbfiles, labelfiles)):
            imgrgb = np.array(Image.open(RGB))
            imgthermal = np.array(Image.open(RGB_T))
            #this ->
            imgthermal = fill_thermal(imgthermal)
            imgthermal = np.expand_dims(imgthermal, axis=-1)
            imglabel = np.array(Image.open(LABEL))
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
            input_imgrgb = resize(input_imgrgb, (*img_shape, 3), mode='reflect')
            input_imgthermal = resize(input_imgthermal, (*img_shape, 1), mode='reflect')
            # input_imglabel = resize(input_imglabel, (*self.img_shape, 1), mode='reflect')
            # Concatenate  images
            input_img = np.concatenate((input_imgrgb, input_imgthermal), axis=-1)
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))

            # Find contours in the segmentation image
            contours, _ = cv.findContours(imglabel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # Iterate through the contours to get bounding boxes
            labels = np.zeros((max_objects, 5))
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
                if i < max_objects:
                    labels[i - num_of_deleted, :] = [object_class - 1, x_center, y_center, bbox_width, bbox_height]
    
            if len(contours) != num_of_deleted:
                #reverse channel order (for saving purposes)
                img_to_save = input_img.transpose(1,2,0)
                rgb_to_save = (img_to_save[:,:,0:3]*255).astype('uint8')
                thermal_to_save = (img_to_save[:,:,3]*255).astype('uint8')
                imsave(save_dir_path+'/rgb/'+Path(RGB).name,rgb_to_save)
                imsave(save_dir_path+'/thermal/'+Path(RGB_T).name,thermal_to_save)
                np.savetxt(save_dir_path+'/labels/'+Path(LABEL).name[:-4]+'.gz',labels)
            
        
if __name__ == '__main__':
    main()