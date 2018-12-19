# README #

This is the repository for camera related algorithm models that are used in Elevator Ads Platform.

## Requirements ##
The following core dependencies need to be installed manually or using Docker:
* CUDA 9
* CuDNN 7 
* Python 3.5+
* OpenCV
* Pytorch >= 0.3.0

The following core depenencies can be installed through `pip3 install -r requirements.txt`:
* Chainer
* Mxnet

## Docker ##
### Installation ###
To use Docker for development, install the following dependencies on host:
* Nvidia Driver
* Docker
* Nvidia-Docker 2

You can also use [this script](https://github.com/pkdogcom/docker-zone/blob/master/setup-host.sh) to setup the host machine automatically.

### Running ###
Start the Docker container using:

```sudo docker run -ti --runtime=nvidia --privileged -e="DISPLAY" -e="QT_X11_NO_MITSHM=1" -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" --ipc=host -p 0.0.0.0:6006:6006 -p 8888:8888 -v /dev/video0:/dev/video0 --name eap-models deepgaze/eap-models-dev bash```

You may also want to add extra `-v` options to map codes/IDE/data into docker container.
 
To restart the container, simply run:

```sudo docker start -i eap-models```

To open multiple docker terminal to the same container, simply run:

```sudo docker exec -ti eap-models bash``` 
 

## Preparation ##
### Gaze Estimation ###
* Download [GazeCapture](http://gazecapture.csail.mit.edu/) dataset
* Untar all tar files in the dataset

### Face Detection ###
* Download pretrained models 

```bash facedet/script/download_models.sh```

### Face Attribute ###
* Download pretrained models

```bash faceattr/script/download_models.sh```

## Training ##
### Gaze Estimation ###
Assuming the GazeCapture dataset is located at `~/fast-storage/GazeCapture`, start training with

```python3 train_gaze.py --root_path ~/fast-storage/GazeCapture --result_path results --dataset gazecapture --model resnet --model_depth 18 --batch_size 1024 --pretrain --log_dir results --n_epochs 50 --lr 2e-5 --n_thread 12 --checkpoint 5```

Check `gaze/opts.py` for more training options.
