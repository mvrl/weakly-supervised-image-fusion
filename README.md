# Weekly Supervised Multi-Image Fusion
This repository contains code and details of the following paper:

M. Usman Rafique, H. Blanton, and N. Jacobs. Weakly super-vised fusion of multiple overhead images.  In the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

The paper has been published here: [Abstract](http://openaccess.thecvf.com/content_CVPRW_2019/html/EarthVision/Rafique_Weakly_Supervised_Fusion_of_Multiple_Overhead_Images_CVPRW_2019_paper.html) [PDF](http://openaccess.thecvf.com/content_CVPRW_2019/papers/EarthVision/Rafique_Weakly_Supervised_Fusion_of_Multiple_Overhead_Images_CVPRW_2019_paper.pdf)

## Using the Code
This repository contains several scripts for training and evaluating models. All the settings are in the file `config.py`. Before running any training or evaluation script, you should specify settings in the config file. Comments explain the purpose of all variables and settings in the `config.py` file.

### Training
To train our proposed model, run the file `train_basemap.py`. This requires the dataset to be set to the one with random clouds and occlusions. This is the default `cfg.data.name` in the config file, as released in the repo. Before running any training script, you need to specify an output directory `cfg.train.out_dir` in `config.py`. The trained model and the loss curves will be saved in this directory.

To train the baseline method, run the file`train_baseline.py` with same dataset i.e. with clouds and occlusions.

If you want to train only the segmentation network, as an upper limit, on clean data, run the file `train_clean_data.py`. This requires dataset to be set to clean images, you can do this by setting `cfg.data.name = 'berlin4x4'`. Again, in the `config.py`, all these options are explained through comments.

### Evaluating a Trained Model
To evaluate a trained model, make sure to set the appropriate folder of the trained model (`cfg.train.out_dir`). To evaluate a trained system, run the file `eval_trained.py`. If a trained model cannot be found, error message will indicate this before exiting.

### Visualizing a Trained Model
By running `visualize_trained.py`, you can save image results on disk, as shown in Figure 6 of the paper. If you want to save segmentation results, run the file `visualize_trained_seg.py`.
## Dataset
As descriped in the paper, we use images and their labels from City-OSM dataset from this paper:

P. Kaiser, J. D. Wegner, A. Lucchi, M. Jaggi, T. Hofmann, and K. Schindler. Learning aerial image segmentation from online maps. IEEE Transactions on Geoscience and Remote Sensing.

You can access their full dataset here: [City-OSM Dataset](https://zenodo.org/record/1154821). Note that we have only presented results by training on images from Berlin and using Potsdam as the test set. As mentioned in their paper, segmentation labels are collected from OSM which may have noise.

To generate partially cloudy images, we have used real cloud images and imposed them on images using Perlin noise as alpha masks. We have used 30 cloud images for training and 10 different cloud images for testing.

### Request Dataset
If you want to replicate our results, without re-collecting data, please send an email to usman dot rafique AT uky dot EDU. We can provide the full dataset, as used in our evaluations, in a single zip file.

## People
Please feel free to contact us for any question or comment.

[M. Usman Rafique](https://usman-rafique.github.io/ "Usman's website")

Hunter Blanton

[Nathan Jacobs](http://cs.uky.edu/~jacobs/ "Nathan's website")

## Permission
The code is provided for academic purposes only without any guarantees. 