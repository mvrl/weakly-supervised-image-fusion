# This file loads the trained model from disk and saves segmentation results on the disk

import torch
import matplotlib.pyplot as plt
import os
from config import cfg
from data_factory import get_dataset
import numpy as np
import torch.nn.functional as F


# This code creates a seg_results folder inside the current folder and saves some images there. If that folder already exists,
# the code throws error on purpose, to avoid overwriting previous results


# Here is a checklist before running this code:
#i. Make sure to select the correct model by setting model_best, if model_best is True the best model wrt val loss,
# will be used. Otherwise, the model saved at the end of training will be used for evaluation.
#ii. Make sure to set the correct folder variable cfg.train.out_dir in config.py


# IMP: select the correct model here:
model_best = True

eval_potsdam = True     # If true, the test set of Potsdam images will be used. If false, the val set from Berlin will
                        # be used

def main():
    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError('The folder does not exist. Make sure to set the correct folder variable cfg.train.out_dir in config.py')

    if os.path.exists(os.path.join(out_dir,'seg_results')):
        raise ValueError('The validation folder image_results already exists. Delete the folder if those results are not needed')
    else:
        os.makedirs(os.path.join(out_dir, 'seg_results'))

    qual_net = torch.load(os.path.join(out_dir, "trained_basemap_checkpoint.pth"))

    print('Network loaded...')
    print(cfg)

    ## getting the dataset
    if eval_potsdam == True:
        cfg.train.mode = 'test_potsdam'
    else:
        cfg.train.mode = 'test'

    ds_test = get_dataset(cfg.train.mode)
    print('Data loaders have been prepared!')

    # loading segmentation net
    segment_net = torch.load(os.path.join(out_dir, "trained_model_checkpoint.pth"))

    qual_net.eval()
    segment_net.eval()

    ctr = 0
    with torch.no_grad():
        for i, data in enumerate(ds_test, 0):
            # reading images
            images = data[0].type('torch.cuda.FloatTensor')

            labels = data[1].type('torch.cuda.LongTensor')  # labels
            # occluded images
            occluded_imgs = data[2]

            # initializing the quality scores of all images
            q_pre = torch.zeros(occluded_imgs[0].shape[0], len(occluded_imgs), occluded_imgs[0].shape[1],
                                occluded_imgs[0].shape[2]).type('torch.cuda.FloatTensor')

            for j in range(len(occluded_imgs)):  # compute all the quality masks
                q_now = qual_net(occluded_imgs[j].type('torch.cuda.FloatTensor'))
                q_pre[:, j, :, :] = q_now[:, 0, :, :]

            # do the softmax across quality masks dimension
            q_final = F.softmax(1 * q_pre, dim=1)

            # make the final basemap
            base_map = 0.0 * occluded_imgs[0].type('torch.cuda.FloatTensor')  # initialization with zero
            for j in range(len(occluded_imgs)):  # compute all the quality masks
                image_now = occluded_imgs[j].type('torch.cuda.FloatTensor')
                base_map = base_map + q_final[:, j, :, :].view(q_now.shape).permute(0, 2, 3, 1) * image_now

            # Computing average as a baseline
            average_image = 0.0 * occluded_imgs[0].type('torch.cuda.FloatTensor') # initialize with zero
            for j in range(len(occluded_imgs)):
                average_image = average_image + occluded_imgs[j].type('torch.cuda.FloatTensor')  # avoiding inline operation i.e. +=

            average_image = average_image / np.float(len(occluded_imgs))

            # segmentation prediction
            predicted = segment_net(images)

            p = torch.argmax(predicted[:, :, :, :], dim=1)

            num_fig = np.minimum(base_map.shape[0], 18)

            plt.ioff()
            for k in range(num_fig):
                # target output
                plt.figure()
                plt.imshow(images[k,: ,:, :].detach().cpu().numpy())
                plt.axis('off')
                fname1 = str(str(ctr) + '_target' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                # basemap
                plt.figure()
                plt.imshow(base_map[k, :, :, :].detach().cpu().numpy())
                plt.axis('off')
                fname1 = str(str(ctr) + '_out_basemap' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                plt.figure()
                plt.imshow(base_map[k, :, :, :].detach().cpu().numpy())
                plt.axis('off')
                fname1 = str(str(ctr) + '_out_basemap' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                # basemap
                plt.figure()
                plt.imshow(average_image[k, :, :, :].detach().cpu().numpy())
                plt.axis('off')
                fname1 = str(str(ctr) + '_out_average' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                # noisy input images
                for j in range(len(occluded_imgs)):
                    plt.figure()
                    plt.imshow(occluded_imgs[j][k,:,:,:])
                    plt.axis('off')
                    fname1 = str(str(ctr) + '_image' +str(j) + '.png')
                    plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                # quality masks
                for j in range(len(occluded_imgs)):
                    plt.figure()
                    plt.imshow(q_final[k, j, :, :].detach().cpu().numpy())
                    plt.axis('off')
                    fname1 = str(str(ctr) + '_mask'+ str(j) + '.png')
                    plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                # segmentation results
                plt.figure()
                plt.imshow(labels[k, :, :].detach().cpu().numpy())
                plt.axis('off')
                fname1 = str(str(ctr) + '_seg_true' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                plt.figure()
                a = p[k, :, :].cpu().detach().numpy()
                plt.axis('off')
                plt.imshow(a)
                fname1 = str(str(ctr) + '_seg_pred' + '.png')
                plt.savefig(os.path.join(out_dir, 'seg_results', fname1), bbox_inches='tight')

                ctr += 1


if __name__ == '__main__':
    main()