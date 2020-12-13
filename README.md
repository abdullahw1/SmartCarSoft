# Sp17_bcs_comsats-Project

## Jetson nano setup

* Download the SD card image from: https://developer.nvidia.com/jetson-nano-sd-card-image

* Flash it in to an SD Card.

* After the initial device setup select language and time zone.

### Installing prerequsities

`sudo apt-get update`

`sudo apt-get upgrade`

`sudo apt-get install git cmake python3-dev nano`

`sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev`

### Configuring your Python environment

`sudo apt-get install python3-pip`

`sudo pip3 install -U pip testresources setuptools`

### Installing deep learning libraries

`sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran`

`sudo apt-get install python3-pip`

`sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11`

`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow`

`sudo apt-get install -y build-essential libatlas-base-dev gfortran`

`sudo pip3 install keras`

`sudo apt-get install libopenblas-base libopenmpi-dev`

`sudo apt-get install python3-pip`

`pip3 install Cython`

`wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl`

`pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl`

`sudo apt-get install libjpeg-dev zlib1g-dev`

`git clone --branch release/0.7 https://github.com/pytorch/vision torchvision`

`cd torchvision`

`sudo python3 setup.py install`

`cd ../`

Finally to install opencv follow this guide:

https://www.jetsonhacks.com/2019/11/22/opencv-4-cuda-on-jetson-nano/

## Dependency installation

`pip install -r requirements.txt`

`cd tiny_yolo/weights/`

`bash download_weights.sh`

## Usage

`python main.py`

## Command line arguments:

`-a`:
Diplaying Momentary Time to Contact without Acceleration (point 2 of research paper)

 _Otherwise it will display with modeling of acceleration which is giving negative results (point 3 of research paper)_

 `python main.py -a`


`-c`: To use camera

`python main.py -c`


`-r`: To use time of 60 fps

*To display TTC without acceleration using camera*

 _Sequence doesn't matter_

python main.py -c -a
