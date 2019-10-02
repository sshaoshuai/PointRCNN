

```shell
pip install tqdm pillow scipy easydict edict tensorboardX pyyaml numba scikit-image pyquaternion sklearn tensorboard==1.14 tensorflow-gpu fire torch==1.0 torchvision==0.2.2.post3 opencv-python==3.4.3.18
```

```shell
conda install tqdm pillow scipy pyyaml numba scikit-image tensorboard opencv
pip install plotly fire easydict edict pyquaternion
```

```shell
tensorboard --logdir=/home/lhr/cdh_workspace/PointRCNN/output/rpn/default/tensorboard
```


```shell
ln -sf /home2/lhr/dataset/lyft_dataset/train_data /home2/lhr/dataset/level5dataset/v1.02-train/v1.02-train

ln -sf /home2/lhr/dataset/lyft_dataset/train_lidar /home2/lhr/dataset/level5dataset/v1.02-train/lidar 

ln -sf /home2/lhr/dataset/lyft_dataset/train_images /home2/lhr/dataset/level5dataset/v1.02-train/images 

ln -sf /home2/lhr/dataset/lyft_dataset/train_maps /home2/lhr/dataset/level5dataset/v1.02-train/maps
```

