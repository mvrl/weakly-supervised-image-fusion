# This file has all the metrics for basemaps as well as segmentation network

# This code computes IoU of three classes (building, road, and background)
# everything is computed in batches using pytorch functions

import torch

def IoU(predicted, labels, extra=False):
	p = torch.argmax(predicted[:, :, :, :], dim=1)

	# IoU of building, channel 0
	pred_build = torch.eq(p, torch.zeros(1).type('torch.cuda.LongTensor'))
	true_build = torch.eq(labels[:, :, :], torch.zeros(1).type('torch.cuda.LongTensor'))

	inter_build = torch.sum(pred_build * true_build)
	union_build = torch.sum(pred_build) + torch.sum(true_build) - inter_build
	iou_build = (inter_build.float()) /(union_build.float() + 1e-6)

	# IoU of road, channel 1
	pred_road = torch.eq(p, torch.ones(1).type('torch.cuda.LongTensor'))
	true_road = torch.eq(labels[:, :, :], torch.ones(1).type('torch.cuda.LongTensor'))

	inter_road = torch.sum(pred_road * true_road)
	union_road = torch.sum(pred_road) + torch.sum(true_road) - inter_road
	iou_road = (inter_road.float()) / (union_road.float() + 1e-6)

	# IoU of background, channel 2
	pred_bg = torch.eq(p, 2 * torch.ones(1).type('torch.cuda.LongTensor'))
	true_bg = torch.eq(labels[:, :, :], 2 * torch.ones(1).type('torch.cuda.LongTensor'))

	inter_bg = torch.sum(pred_bg * true_bg)
	union_bg = torch.sum(pred_bg) + torch.sum(true_bg) - inter_bg
	iou_bg = (inter_bg.float()) / (union_bg.float()  + 1e-6)

	# mean IoU
	if extra==True:
		mIoU = (iou_bg + iou_road + iou_build ) / 3.0  # mean IoU

		# frequency weighted IoU
		total_pix = torch.sum(true_build) + torch.sum(true_road) + torch.sum(true_bg)
		fwIoU = (iou_bg*torch.sum(true_bg) + iou_road*torch.sum(true_road) + iou_build*torch.sum(true_build) ) / total_pix.float()

		# pixel accuracy
		acc = ( inter_road + inter_build + inter_bg ) / total_pix.float()


		if total_pix == 0:
			print('Oops, total pix = 0')
			print('debug')
		return iou_build, iou_road, iou_bg, mIoU, fwIoU, acc

	return iou_build, iou_road, iou_bg
