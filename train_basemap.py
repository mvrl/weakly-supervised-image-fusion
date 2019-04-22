# Main file that trains a quality net and a segmentation net.
# Our quality net synthesizes a fused image. This fused image is fed into the segmentation network.


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import cfg
from data_factory import get_dataset
import numpy as np
from datetime import datetime
import os
from My_Unet import Net_lighter as Unet_class   # light UNet quarter the channels
from My_Unet import Q_Net_lighter as QualityNet_class   # light UNet quarter the channels
from metrics import IoU

def main():
    # setting output directory
    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print('Folder already exists. Are you sure you want to overwrite results?')
        print('Debug')  # put a break point here

    print('Configuration:')
    print(cfg)

    ## Getting the dataset
    # Training data loader
    cfg.train.mode = 'train'
    ds_train = get_dataset( cfg.train.mode)

    # validation data loader
    cfg.train.mode = 'test'
    ds_test = get_dataset(cfg.train.mode)
    print('Data loaders have been prepared!')

    ## Getting the model
    cfg.train.mode = 'train'

    # segmentation network
    net = Unet_class()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # quality network
    qual_net = QualityNet_class()
    qual_net.to(device)

    print('Network loaded. Starting training...')

    # weights for  # building, road, BG
    my_weight = torch.from_numpy(np.asarray([1, 2, 0.5])).type('torch.cuda.FloatTensor')
    criterion = torch.nn.CrossEntropyLoss(weight=my_weight )

    l1_maps = torch.nn.L1Loss(reduction='sum')

    param = list(net.parameters()) + list(qual_net.parameters())
    optim = torch.optim.Adam(param, lr=cfg.train.learning_rate, weight_decay=cfg.train.learning_rate_decay)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    ep = cfg.train.num_epochs # number of epochs

    # loss logs
    loss_train = 9999.0*np.ones(ep)
    temp_train_loss = 0
    loss_val = 9999.0*np.ones(ep)

    # training the network
    for epoch in range(ep):
        running_loss = 0.0
        running_ctr = 0

        # switch model to training mode, clear gradient accumulators
        net.train()
        qual_net.train()
        optim.zero_grad()

        scheduler.step()  # update learning rate

        t1 = datetime.now()
        for i, data in enumerate(ds_train, 0):
            optim.zero_grad()

            # reading images
            images = data[0].type('torch.cuda.FloatTensor')

            # labels
            labels = data[1].type('torch.cuda.LongTensor')

            # occluded images
            occluded_imgs = data[2]

            # initializing the quality scores of all images
            q_pre = torch.zeros(occluded_imgs[0].shape[0], len(occluded_imgs), occluded_imgs[0].shape[1],
                                occluded_imgs[0].shape[2]).type('torch.cuda.FloatTensor')

            for j in range(len(occluded_imgs)):  # compute all the quality masks
                q_now = qual_net(occluded_imgs[j].type('torch.cuda.FloatTensor'))
                q_pre[:, j, :, :] = q_now[:, 0, :, :]

            # do the softmax across quality masks dimension
            q_final = F.softmax(q_pre, dim=1)

            # make the final basemap
            base_map = 0.0 * occluded_imgs[0].type('torch.cuda.FloatTensor')  # initialization with zero
            for j in range(len(occluded_imgs)):  # synthesizing fused image by combining images, weighted by quality scores
                image_now = occluded_imgs[j].type('torch.cuda.FloatTensor')
                base_map = base_map + q_final[:, j, :, :].view(q_now.shape).permute(0, 2, 3, 1) * image_now

            predicted = net(base_map)

            loss = criterion(predicted, labels)

            loss.backward()

            optim.step()

            # print statistics
            running_loss += loss.item()
            running_ctr += 1
            if i %25 ==0:
                t2 = datetime.now()
                delta = t2 - t1
                t_print = delta.total_seconds()
                temp_train_loss = running_loss/25.0
                print('[%d, %5d out of %5d] loss: %f, time = %f' %
                      (epoch + 1, i + 1, len(ds_train) , running_loss / running_ctr,  t_print ))

                iou_build, iou_road, iou_bg = IoU(predicted, labels)
                print('building IoU = ' + str(iou_build) + ', road IoU = ' + str(iou_road) + ', background IoU = ' + str(iou_bg) )

                basemap_error = l1_maps(base_map, images)
                print('L1 error (base map, true image) = ' + str(basemap_error.item()))
                running_loss = 0.0
                running_ctr = 0
                t1 = t2

        # at the end of every epoch, calculating val loss
        net.eval()
        qual_net.eval()
        val_loss = 0

        with torch.no_grad():
            for i, data in enumerate(ds_test, 0):
                # get images
                images = data[0].type('torch.cuda.FloatTensor')

                # labels
                labels = data[1].type('torch.cuda.LongTensor')

                # occluded images
                occluded_imgs = data[2]

                # initializing the quality scores of all images
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

                predicted = net(base_map)

                loss = criterion(predicted, labels)

                # val loss
                val_loss +=  loss.item()

            # print statistics
            val_loss = val_loss /len(ds_test)
            print('End of epoch ' + str(epoch + 1) + '. Val loss is ' + str(val_loss))

            print('Following stats are only for the last batch of the test set:')
            iou_build, iou_road, iou_bg = IoU(predicted, labels)
            print('building IoU = ' + str(iou_build) + ', road IoU = ' + str(iou_road) + ', background IoU = ' + str(
                iou_bg))
            basemap_error = l1_maps(base_map, images)
            print('L1 error (base map, true image) = ' + str(basemap_error.item()))

            # Model check point
            if val_loss < np.min(loss_val, axis=0):
               model_path = os.path.join(out_dir, "trained_model_checkpoint.pth")
               torch.save(net, model_path)   # segmentation network

               model_path = os.path.join(out_dir, "trained_basemap_checkpoint.pth")
               torch.save(qual_net, model_path)   # basemap network
               print('Model saved at epoch ' + str(epoch+1))


            # saving losses
            loss_val[epoch] = val_loss
            loss_train[epoch] = temp_train_loss

            temp_train_loss = 0  # setting additive losses to zero


    print('Training finished')
    # saving model
    model_path = os.path.join(out_dir, "trained_model_end.pth")
    torch.save(net, model_path)

    # Saving logs
    log_name = os.path.join(out_dir, "logging.txt")
    with open(log_name, 'w') as result_file:
        result_file.write('Logging... \n')
        result_file.write('Validation loss ')
        result_file.write(str(loss_val))
        result_file.write('\nTraining loss  ')
        result_file.write(str(loss_train))

    print('Model saved')

    # saving loss curves
    a = loss_val
    b = loss_train
    print(a[0:epoch])

    plt.figure()
    plt.plot(b[0:epoch])
    plt.plot(a[0:epoch])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss', 'Validation Loss'])
    fname1 = str('loss.png')
    plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')

    print('All done!')

if __name__ == '__main__':
    main()