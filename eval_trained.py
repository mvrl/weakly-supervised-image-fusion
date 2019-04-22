# This file loads the trained model from disk and evaluates the trained model TEN times and computes the average results.
import torch
import os
from config import cfg
from data_factory import get_dataset
import numpy as np
import torch.nn.functional as F
from metrics import IoU


# Here is a checklist before running evaluation
#i. Make sure to select the correct model by setting model_best, if model_best is True the best model wrt val loss,
# will be used. Otherwise, the model saved at the end of training will be used for evaluation.
#ii. Make sure to set the correct folder variable cfg.train.out_dir in config.py

# IMP: select the correct model here:
model_best = True       # if true, the model with lowest validation loss will be used

repeat_times = 10       # Compute evaluation this many times and then compute average

eval_baseline = False    # CRITICAL: set this to true if you are evaluating baseline. This MUST be FALSE if the proposed
                        # basemap method is being evaluated

eval_potsdam  = True            # Evaluate on Potsdam. This is the Test set. If this is set to false, the val set from
                                # Berlin will be used for evaluation

def main():
    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError('The folder does not exist. Make sure to set the correct folder variable cfg.train.out_dir in config.py')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if eval_baseline == False:      # quality net based method

        qual_net = torch.load(os.path.join(out_dir, "trained_basemap_checkpoint.pth")) # loading quality net
        qual_net.eval()

        qual_net.to(device)

    # loading segmentation net
    segment_net = torch.load(os.path.join(out_dir, "trained_model_checkpoint.pth"))
    segment_net.eval()
    segment_net.to(device)

    print('Network loaded...')
    print(cfg)

    ## getting the dataset
    if eval_potsdam == True:
        cfg.train.mode = 'test_potsdam'         # Potsdam
    else:
        cfg.train.mode = 'test'                 # Val set of Berlin

    ds_test = get_dataset(cfg.train.mode)
    print('Data loaders have been prepared!')

    # Metrics for pixel-wise comparison of fused images to respective clean images
    l1_error_abs = torch.nn.L1Loss(reduction='sum')     # This is dependent on image size
    l1_error_mean = torch.nn.L1Loss(reduction='mean')   # This is reported in the paper

    # Initializae metrics
    abs_error = 0
    mean_error = 0

    iou_build = 0
    iou_road = 0
    iou_bg = 0
    mIoU = 0
    fwIou = 0
    acc = 0


    with torch.no_grad():
        for t in range(repeat_times):  # evaluate everything 10 times

            for i, data in enumerate(ds_test, 0):
                images = data[0].type('torch.cuda.FloatTensor') # reading images

                # labels
                labels = data[1].type('torch.cuda.LongTensor')
                occluded_imgs = data[2]


                if eval_baseline == False:
                    q_pre = torch.zeros(occluded_imgs[0].shape[0], len(occluded_imgs), occluded_imgs[0].shape[1],
                                        occluded_imgs[0].shape[2]).type('torch.cuda.FloatTensor')

                    for j in range(len(occluded_imgs)):  # compute all the quality masks
                        q_now = qual_net(occluded_imgs[j].type('torch.cuda.FloatTensor'))
                        q_pre[:, j, :, :] = q_now[:, 0, :, :]

                    # do the softmax across quality masks dimension
                    q_final = F.softmax(q_pre, dim=1)

                    # make the final basemap
                    base_map = 0.0 * occluded_imgs[0].type('torch.cuda.FloatTensor')  # initialization with zero
                    for j in range(len(occluded_imgs)): # synthesizing fused image by combining images, weighted by quality scores
                        image_now = occluded_imgs[j].type('torch.cuda.FloatTensor')
                        base_map = base_map + q_final[:, j, :, :].view(q_now.shape).permute(0, 2, 3, 1) * image_now

                if eval_baseline == True:      # Evaluating baseline?
                    # the following code is for Baseline (average) ONLY
                    base_map = 0.0 * occluded_imgs[0].type('torch.cuda.FloatTensor')  # initialize with zero
                    for j in range(len(occluded_imgs)):
                        base_map = base_map + occluded_imgs[j].type(
                            'torch.cuda.FloatTensor')  # avoiding inline operation i.e. +=

                        base_map = base_map / np.float(len(occluded_imgs))

                loss_abs = l1_error_abs(base_map, images)
                loss_mean = l1_error_mean(base_map, images)

                abs_error += loss_abs.item()
                mean_error += loss_mean.item()

                # segmentation performance
                predicted = segment_net(base_map)
                i1, i2, i3, i4, i5, i6 = IoU(predicted, labels, extra=True)
                iou_build += i1
                iou_road += i2
                iou_bg += i3
                mIoU += i4
                fwIou += i5
                acc += i6

            print('Completed ' + str(t) + 'out of ' + str(repeat_times))


    # computing average
    abs_error /= (len(ds_test) * repeat_times)
    mean_error /= (len(ds_test) * repeat_times)

    # average of segmentation numbers
    iou_build /= ( len(ds_test)  * repeat_times)
    iou_road /= (len(ds_test)  * repeat_times)
    iou_bg /= (len(ds_test) * repeat_times)
    mIoU /= (len(ds_test)  * repeat_times)
    fwIou /= (len(ds_test)  * repeat_times)

    acc /= (len(ds_test)  * repeat_times)

    print('Mean error on test set = ' + str(mean_error))
    print('Absolute error on test set = ' + str(abs_error))
    print('Building IoU on test set = ' + str(iou_build))
    print('Road IoU on test set = ' + str(iou_road))
    print('BG IoU on test set = ' + str(iou_bg))
    print('mIoU on test set = ' + str(mIoU))
    print('Frequency weighted IoU on test set = ' + str(fwIou))
    print('Pixel accuracy on test set = ' + str(acc))

    if eval_potsdam == True:
        if eval_baseline:
            n1 = str('eval_result_Potsdam_baseline_multiple.txt')
            fname = os.path.join(out_dir, n1)
        else:
            n1 = str('eval_result_Potsdam_multiple.txt')
            fname = os.path.join(out_dir, n1)
    else:
        fname = os.path.join(out_dir, 'eval_result_Berlin.txt')

    # saving results on disk
    with open(fname, 'w') as result_file:
        result_file.write('Logging... \n')
        result_file.write('Mean error on test set =  ')
        result_file.write(str(mean_error))
        result_file.write('\nAbsolute error on test set =   ')
        result_file.write(str(abs_error))
        result_file.write('\nBuilding IoU on test set =   ')
        result_file.write(str(iou_build))
        result_file.write('\nRoad IoU on test set =   ')
        result_file.write(str(iou_road))
        result_file.write('\nBG IoU on test set =   ')
        result_file.write(str(iou_bg))
        result_file.write('\nMean IoU on test set =   ')
        result_file.write(str(mIoU))
        result_file.write('\nfrequency weighted IoU on test set =   ')
        result_file.write(str(fwIou))
        result_file.write('\nPixel accuracy on test set =   ')
        result_file.write(str(acc))

    print('All done. Results saved in eval_result.txt in output directory')

if __name__ == '__main__':
    main()