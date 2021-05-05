## New!!!
We have released a PyTorch implementation of the method for relative camera pose estimation. The code and pre-trained models are available at https://github.com/AaltoVision/RelPoseNet

# Relative camera pose estimation using CNNs
Torch code and models for _Relative Camera Pose Estimation Using Convolutional Neural Networks_

https://arxiv.org/abs/1702.01381

# Running the code
* First, you need to download original DTU dataset (136Gb) http://roboimagedata.compute.dtu.dk/. It can be done by using the following command:
```
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Cleaned.zip
```
* Inside `pre-trained/` folder run `download_models.sh` script downloading pre-trained HybridCNN (http://places.csail.mit.edu/) model. It is needed only for training the proposed model
* And finally
```
th main.lua -do_evaluation -source_image_path <path/to/DTU/Cleaned> -weights ./pre-trained/siam_hybridnet_fullsized.t7
```

# Bibtex
```
@inproceedings{Melekhov2017relativePoseCnn,
    author = {Iaroslav Melekhov and Juha Ylioinas and Juho Kannala and Esa Rahtu},
    title = {Relative Camera Pose Estimation Using Convolutional Neural Networks},
    url = {https://arxiv.org/abs/1702.01381},
    year = {2017}}
```
