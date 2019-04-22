# All the dataloaders are implemented in this file.

import torch
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torch.utils.data import Dataset
from skimage import transform as sk_transform
from skimage.io import imread
import os
from config import cfg
import csv



# Source of Perlin noise code:
# https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
def generate_perlin_noise_2d(shape, res):
    # shape: shape of the generated array (tuple of 2 ints)
    # res: number of periods of noise to generate along each axis (tuple of 2 ints)

    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Fixing some errors - edit by Usman Feb 26, 2019
    grid = grid[:shape[0], :shape[1], :]

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    final_perlin = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    final_perlin += 1
    return final_perlin


###############
## Clean data
##############
class Dataset_Berlin4x4(Dataset):    # images split in 4x4 grid, resized to 300x300 pixels
    def __init__(self, csv_file, root_dir, data_dir, using_onehot):

        self.rows = []
        with open(csv_file, 'r', encoding='utf-8-sig') as myfile:
            reader = csv.reader(myfile)
            for r in reader:
                self.rows.append(r)

        self.root_dir = os.path.join(root_dir, data_dir)
        self.using_onehot = using_onehot

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        crop_loc_raw = int(self.rows[idx][2])

        # Reading aerial image
        img_name = os.path.join(self.root_dir, self.rows[idx][0])
        img_big = imread(img_name)/ 255.0
        img_big = img_big[0:cfg.data.image_size_full[0], 0:cfg.data.image_size_full[1], :]  # cropping

        label_name = os.path.join(self.root_dir, self.rows[idx][1])
        label_big = imread(label_name)
        label_big = label_big[0:cfg.data.image_size_full[0], 0:cfg.data.image_size_full[1], :]  # cropping

        # finding 2D grid location
        grid_col = int(crop_loc_raw) % 4
        grid_row = int(np.floor(crop_loc_raw / 4))

        # Splitting image
        ps = int(cfg.data.image_size_full[0]/4)
        img_single = img_big[grid_row * ps:(grid_row + 1) * ps, grid_col * ps:(grid_col + 1) * ps, :]

        # raw label
        label_single = label_big[grid_row * ps:(grid_row + 1) * ps, grid_col * ps:(grid_col + 1) * ps, :]
        # splitting label

        # Converting labels to one-hot
        label_one_hot = 0.0 * img_single  # 0: building red, 1 road blue, 2 BG white

        # building channel 0
        label_one_hot[:, :, 0] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 2], 0)))

        # road channel 1
        label_one_hot[:, :, 1] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 0), np.equal(label_single[:, :, 2], 255)))

        # background, channel 2
        label_one_hot[:, :, 2] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 1], 255)))

        label_one_hot[:, :, 2] = 1 * np.logical_and(label_one_hot[:, :, 2], np.equal(label_single[:, :, 2], 255))

        # fixing some noisy, left-out pixels, assigning them to BG . These are the ones ==0 in all 3 channels
        all_zeros = np.logical_and(np.equal(label_one_hot[:, :, 0], 0), np.equal(label_one_hot[:, :, 1], 0))
        all_zeros = np.logical_and(all_zeros, np.equal(label_one_hot[:, :, 2], 0))

        label_one_hot[:, :, 2] += 1 * all_zeros   # add these noisy pixels to background

        # resizing
        label_one_hot = sk_transform.resize(label_one_hot, (300, 300), preserve_range=True)
        img_single = sk_transform.resize(img_single, (300, 300), preserve_range=True)

        if not self.using_onehot:
            label_one_hot = np.argmax(label_one_hot, 2)

        return img_single, label_one_hot


#######################################################
#### Berlin real clouds, random location and Occlusions
#######################################################

class dataset_Four_Berlin_realClouds_n_Occ(Dataset):        # images split in 4x4 grid, resized to 300x300 pixels

    def __init__(self, csv_file, root_dir, data_dir, cloud_file, cloud_dir, using_onehot, area_ratio, num_images = 4):

        self.rows = []
        with open(csv_file, 'r', encoding='utf-8-sig') as myfile:
            reader = csv.reader(myfile)
            for r in reader:
                self.rows.append(r)

        self.root_dir = os.path.join(root_dir, data_dir)

        self.using_onehot = using_onehot

        # reading cloud images CVS file
        cloud_rows = []
        with open(cloud_file, 'r', encoding='utf-8-sig') as myfile:
            reader = csv.reader(myfile)
            for r in reader:
                cloud_rows.append(r)

        self.num_cloud_imgs = len(cloud_rows)  # number of images of clouds

        # reading cloud images
        self.cloud_imgs = []
        for j in range(self.num_cloud_imgs):
            img_name = os.path.join(root_dir, cloud_dir, cloud_rows[j][0])
            self.cloud_imgs.append(imread(img_name)/255.0)

        self.num_occluded = num_images

        self.area_ratio = area_ratio

        # Making permanent masks, same dize as that images i.e. 300x300
        # these masks are then applied for alpha blending with cloud images. The masks are later resized to the size
        # of cloud
        mask= np.zeros((int(300), int(300)))
        mask2 = np.zeros((int(300), int(300)))

        center_x = int(300 / 2)
        center_y = int(300 / 2)

        rad = 150

        # cut off till values are 1
        cut_off = 0.5

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                d = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                d_r = d / rad
                if d_r < (cut_off):  # 0.5
                    mask[i, j] = 1
                    # self.mask3[i, j] = 1
                elif d_r < (0.6):  # 0.5
                    new_d = (d_r - 0.5) / 0.1
                    mask[i, j] = 1 + (- 0.2 * new_d)
                elif d_r < 0.65:
                    mask[i, j] = 0.8  # 1 + (- 0.1 * new_d)
                else:
                    d_new = ((d_r - 0.65)) / (1 - 0.65)
                    mask2[i, j] = 0.8 - (d_new * 0.8)

        mask = np.clip(mask, a_min=0, a_max=1)
        mask2 = np.clip(mask2, a_min=0, a_max=1)

        # A list of final masks
        self.list_final_masks = []
        perlin_block = 5

        # number of ready-to-go masks
        self.num_masks = 30

        for j in range(self.num_masks):
            perlin_noise = generate_perlin_noise_2d((mask.shape[0], mask.shape[1]),
                                                    (perlin_block, perlin_block))
            perlin_noise = np.clip(perlin_noise, a_min=0, a_max=1)

            # complete mask
            m = perlin_noise*mask2 + mask
            m = gaussian_filter(m, sigma=(5, 5))
            self.list_final_masks.append(m) #*perlin_noise2


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        crop_loc_raw = int(self.rows[idx][2])

        # Reading full aerial image
        img_name = os.path.join(self.root_dir, self.rows[idx][0])
        img_big = imread(img_name)/ 255.0
        img_big = img_big[0:cfg.data.image_size_full[0], 0:cfg.data.image_size_full[0], :]  # cropping

        # reading segmentation labels
        label_name = os.path.join(self.root_dir, self.rows[idx][1])
        label_big = imread(label_name)
        label_big = label_big[0:cfg.data.image_size_full[0], 0:cfg.data.image_size_full[0], :]  # cropping

        # finding 2D grid location
        grid_col = int(crop_loc_raw) % 4
        grid_row = int(np.floor(crop_loc_raw / 4))

        # Splitting image
        ps = int(cfg.data.image_size_full[0]/4)
        img_single = img_big[grid_row * ps:(grid_row + 1) * ps, grid_col * ps:(grid_col + 1) * ps, :]

        # splitting label
        label_single = label_big[grid_row * ps:(grid_row + 1) * ps, grid_col * ps:(grid_col + 1) * ps, :]


        ## Preparing label
        # Building = red     = [255,0,0], Road = blue    = [0,0,255] , Background = white = [255,255,255]
        label_one_hot = 0.0 * img_single  # 0: building red, 1 road blue, 2 BG white

        # building channel 0
        label_one_hot[:, :, 0] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 2], 0)))

        # road channel 1
        label_one_hot[:, :, 1] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 0), np.equal(label_single[:, :, 2], 255)))

        # background, channel 2
        label_one_hot[:, :, 2] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 1], 255)))

        label_one_hot[:, :, 2] = 1 * np.logical_and(label_one_hot[:, :, 2], np.equal(label_single[:, :, 2], 255))

        # fixing some noisy, left-out pixels, assigning them to BG . These are the ones ==0 in all 3 channels
        all_zeros = np.logical_and(np.equal(label_one_hot[:, :, 0], 0), np.equal(label_one_hot[:, :, 1], 0))
        all_zeros = np.logical_and(all_zeros, np.equal(label_one_hot[:, :, 2], 0))

        label_one_hot[:, :, 2] += 1 * all_zeros   # add these to

        if not self.using_onehot:
            label_one_hot = np.argmax(label_one_hot, 2)

        # reducing size of images and labels
        img_single = sk_transform.resize(img_single, (300, 300), preserve_range=True )
        label_one_hot = sk_transform.resize(label_one_hot, (300, 300), preserve_range=True )

        #####
        ## Clouds
        ####
        # Randomly pick a cloud image
        img_index = np.random.randint(low = 0, high=self.num_cloud_imgs)
        cloud_img = self.cloud_imgs[img_index]

        occluded_imgs = []
        area = self.area_ratio * img_single.shape[0] * img_single.shape[1]

        for j in range(self.num_occluded):          # Generate this many images
            if j%2==0:          # Odd: CLOUD
                # Augmentation on cloud image
                seq = np.random.randint(0, 5)
                if seq == 0:  # vertical flip
                    cloud_img = np.flip(cloud_img, axis=0)

                elif seq == 1:  # horizontal flip
                    cloud_img = np.flip(cloud_img, axis=1)

                elif seq == 2:  # both flipp
                    cloud_img2 = np.flip(cloud_img, axis=1)
                    cloud_img = np.flip(cloud_img2, axis=0)

                elif seq == 3:  # rotate 90 deg
                    cloud_img = np.rot90(cloud_img, k=1)

                elif seq == 4:  # rotate 90x3 deg
                    cloud_img = np.rot90(cloud_img, k=3)

                alpha_1 = np.zeros((int(label_one_hot.shape[0]), int(label_one_hot.shape[1]), 1))

                # max height and width
                h_max = img_single.shape[0]
                w_max = img_single.shape[1]

                # size
                aspect_ratio = np.random.uniform(low=0.5, high=2)

                w = int(np.sqrt(area/aspect_ratio))
                h = int(area/w)

                w = np.minimum(w, w_max - 2)
                h = np.minimum(h, h_max - 2)

                # making sure h and w are even numbers
                if w%2 ==1:     # if odd
                    w += 1
                if h%2==1:
                    h += 1

                # randomly generate location of clouds
                center_x = np.random.randint(low=int((w) / 2), high=w_max - int(w / 2))
                center_y = np.random.randint(low=int((h) / 2), high=h_max - int(h / 2))

                #Reading the cloud mask
                selected_mask = np.random.randint(low=0, high=self.num_masks)   # random selection from prepared masks
                mask = self.list_final_masks[selected_mask]

                # Apply augmentation on the mask
                seq2 = np.random.randint(0, 6)
                if seq2 == 0:  # vertical flip
                    mask = np.flip(mask, axis=0)

                elif seq2 == 1:  # horizontal flip
                    mask = np.flip(mask, axis=1)

                elif seq2 == 2:  # both flipp
                    mask2 = np.flip(mask, axis=1)
                    mask = np.flip(mask2, axis=0)

                elif seq2 == 3:  # rotate 90 deg
                    mask = np.rot90(mask, k=1)

                elif seq2 == 4:  # rotate 90x3 deg
                    mask = np.rot90(mask, k=3)

                mask_now = sk_transform.resize(mask, (h,w), preserve_range=True)
                alpha_1[center_y - int(h / 2):center_y + int(h / 2), center_x - int(w / 2):center_x + int(w / 2), 0] = mask_now

                # Random rotation
                rot_angle = np.random.randint(low=-45, high=45)
                alpha_1 = sk_transform.rotate(alpha_1, rot_angle)

                # Alpha blending
                c1 = (alpha_1) * cloud_img + (1 - alpha_1) * img_single

            else:           # Even: OCCLUSION
                if self.area_ratio > 0.51:
                    ## Image more than half is occluded, either horizontally or verticaly
                    horiz_ax = np.random.randint(low=0, high=2)
                    end_p = img_single.shape[0]  # this assumes image is square

                    ## horizontal splitting
                    if horiz_ax == 1:
                        top_occluded = np.random.randint(low=0, high=2)  # which side to occlude
                        h_max = img_single.shape[0]
                        w_max = img_single.shape[1]
                        h1 = np.random.randint(low=int(end_p / 2), high=end_p - 5)
                        area_zero_offset = h1 * w_max
                        h2_offset = (area - area_zero_offset) * 2 / w_max
                        h2 = h1 + h2_offset
                        h2 = np.minimum(h2, end_p - 2)

                        if top_occluded == 0:  # bottom occlusion
                            h1 = h_max - h1
                            h2 = h_max - h2

                        v1 = np.asarray([w_max, h2 - h1])  #
                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        positions[0, :] -= (w_max) / 2  #
                        positions[1, :] -= (h1 + h2) / 2  #
                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        if top_occluded == 1:  # top occluded
                            mask1 = 1 * np.less_equal(ans_c, 0)
                            c1 = mask1 * img_single

                        else:  # occlude the bottom
                            mask1 = 1 * np.greater_equal(ans_c, 0)
                            c1 = mask1 * img_single

                    else:
                        ##  vertical splitting
                        left_occluded = np.random.randint(low=0, high=2)  # which side to occlude

                        h_max = img_single.shape[0]
                        w_max = img_single.shape[1]
                        w1 = np.random.randint(low=int(end_p / 2), high=end_p - 5)
                        area_zero_offset = w1 * w_max
                        w2_offset = (area - area_zero_offset) * 2 / w_max
                        w2 = w1 + w2_offset
                        w2 = np.minimum(w2, end_p - 2)

                        if left_occluded == 0:  # bottom occlusion
                            w1 = w_max - w1
                            w2 = w_max - w2

                        v1 = np.asarray([w2 - w1, h_max])
                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        positions[0, :] -= (w1 + w2) / 2
                        positions[1, :] -= (h_max) / 2
                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        if left_occluded == 1:  # left side occluded
                            mask1 = 1 * np.greater_equal(ans_c, 0)
                            c1 = mask1 * img_single

                        else:  # occlude right side
                            mask1 = 1 * np.less_equal(ans_c, 0)
                            c1 = mask1 * img_single

                else:
                    ## If area less than half, diagonal clipping at any location
                    occ_loc = np.random.randint(low=0, high=4)  # 0=top left, 1=top right, 2= bottom left, 3 = bottom right
                    # computing area
                    aspect_ratio = np.random.uniform(low=0.5, high=2)
                    w = int(np.sqrt(2*area / aspect_ratio))
                    h = int(area / w)
                    # max height and width
                    h_max = img_single.shape[0]
                    w_max = img_single.shape[1]
                    w = np.minimum(w, w_max - 2)
                    h = np.minimum(h, h_max - 2)

                    end_p = img_single.shape[0]  # this assumes image is square
                    if occ_loc==0:      # top left
                        v1 = np.asarray([w, -h])  # Vector from first point to the second one

                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])

                        # vector from every point to the midpoint of the vector [a, -b]
                        positions[0, :] -= w / 2
                        positions[1, :] -= h / 2

                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        mask1 = 1 * np.less_equal(ans_c, 0)
                        c1 = mask1 * img_single

                    elif occ_loc == 1:  # top right
                        w = end_p - w  # now measuring the offset from the edges

                        v1 = np.asarray([w - end_p, -h])

                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])

                        positions[0, :] -= (end_p + w) / 2
                        positions[1, :] -= (h) / 2

                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        mask1 = 1 * np.greater_equal(ans_c, 0)
                        c1 = mask1 * img_single

                    elif occ_loc == 2:  # bottom left
                        h = end_p - h           # now measuring the offset from the edges
                        v1 = np.asarray([w , end_p - h])

                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        positions[0, :] -= ( w) / 2
                        positions[1, :] -= (end_p + h) / 2

                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        mask1 = 1 * np.greater_equal(ans_c, 0)
                        c1 = mask1 * img_single

                    elif occ_loc == 3:  # bottom right
                        h = end_p - h   # now measuring the offset from the edges
                        w = end_p - w

                        v1 = np.asarray([w - end_p, end_p - h])

                        l1 = (np.arange(300)).astype(np.float64)
                        X, Y = np.meshgrid(l1, l1)
                        positions = np.vstack([X.ravel(), Y.ravel()])

                        positions[0, :] -= (end_p + w) / 2  #
                        positions[1, :] -= (end_p + h) / 2  #

                        ans_c = np.cross(positions, v1, axisa=0, axisb=0)
                        ans_c = np.reshape(ans_c, (300, 300, 1))

                        mask1 = 1 * np.less_equal(ans_c, 0)
                        c1 = mask1 * img_single

            occluded_imgs.append(c1)

        return img_single, label_one_hot, occluded_imgs


def get_dataset(mode):
    # Get dataset object by its name and mode (train/test)

    data_folder = cfg.data.root_dir     # set data directory

    ## Clean images, to train upper bound
    if cfg.data.name == 'berlin4x4':    # clean images and labels
        # select CSV file for training/test set
        data_dir = 'berlin'
        if mode == 'train':
            split_file = os.path.join(data_folder, data_dir,  'split_files', "berlin_train.csv")  #
        elif mode == 'test_potsdam':
            split_file = os.path.join(data_folder, 'postdam', 'split_files', 'postdam_all.csv' )
            data_dir = 'postdam'
        elif mode == 'test':
            split_file = os.path.join(data_folder, data_dir,  'split_files', "berlin_test.csv")
        else:
            raise ValueError("Mode {} is unknown".format(mode))

        ds = Dataset_Berlin4x4(split_file, data_folder, data_dir, using_onehot= False)

    ##  To generate synthetic dataset with real cloud images and occlusions
    elif cfg.data.name == 'Four_Berlin_realClouds_random_occ':   #
        # select CSV file for training/test set
        data_dir = 'berlin'
        if mode == 'train':         # Train set, from Berlin
            split_file = os.path.join(data_folder, data_dir, 'split_files', "berlin_train.csv")  #
            cloud_file = os.path.join(data_folder, 'clouds', 'clouds_list_train.csv')
        elif mode == 'test':         # Val set, from Berlin
            split_file = os.path.join(data_folder, data_dir, 'split_files', "berlin_val.csv")
            cloud_file = os.path.join(data_folder, 'clouds', 'clouds_list_test.csv')
        elif mode == 'test_potsdam':   # Test set, from Potsdam
            split_file = os.path.join(data_folder, 'potsdam', 'split_files', 'potsdam_all.csv')
            data_dir = 'potsdam'
            cloud_file = os.path.join(data_folder, 'clouds', 'clouds_list_test.csv')
        else:
            raise ValueError("Mode {} is unknown".format(mode))

        cloud_dir = 'clouds'
        ds = dataset_Four_Berlin_realClouds_n_Occ(split_file, data_folder, data_dir, cloud_file, cloud_dir,
                        using_onehot=False, num_images = cfg.data.num_images, area_ratio = cfg.data.area_fraction)


    # preparing pytorch data loader
    ds_final = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle, num_workers=cfg.train.num_workers)

    return ds_final