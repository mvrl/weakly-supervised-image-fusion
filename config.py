# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

## MODEL
cfg.model = edict()

# at this momenet, just passing the name here, not using
cfg.model.name = 'net_Unet'

## DATA
cfg.data = edict()
cfg.data.name = 'Four_Berlin_realClouds_random_occ'  # real clouds at random locations plus occlusions
                                    #'berlin4x4'   # clean Berlin images and labels. Each big image is split in 4 images


cfg.data.splits = (4, 4)
cfg.data.image_size_full = (2208, 2208) #read this portion for uniform size and to be divisible by 16

cfg.data.num_images = 2                 # how many cloudy and occluded images to generate
cfg.data.area_fraction = 0.5            # what portion of the image should have clouds or occlusions

cfg.data.root_dir = 'C:/Usman/Datasets/Image_Fusion'   #Usman's machine


## Training details
cfg.train = edict()
cfg.train.mode = 'train'                   # training mode ('test'|'train')
cfg.train.batch_size =  5                  # batch size
cfg.train.shuffle = True               # shuffle training samples
cfg.train.num_epochs = 20                # number of training epochs  ...
cfg.train.num_workers = 5                # workers for data loading
cfg.train.learning_rate = 1e-4           # initial learning rate for adam.
cfg.train.learning_rate_decay = 4.0*1e-5          # initial learning rate for adam.

cfg.train.out_dir = './outputs/2_imgs2_50area' # Set the name of directory. Trained model will be saved here and
                                               # evaluation code will save results in this directory
