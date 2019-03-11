This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use native installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).


### Usage
1. Install dependencies
```bash
$ sudo apt-get install libprotobuf-dev libprotoc-dev ros-kinetic-dbw-mkz-msgs
$ sudo pip install eventlet==0.19.0
$ sudo pip install Flask==0.11.1
$ sudo pip install python-socketio==1.6.1
$ sudo pip install attrdict==2.0.0
$ sudo pip install utils
$ sudo pip install tensorflow-gpu==1.13.1
```

2. Install protobuf-compiler
```bash
$ wget https://github.com/protocolbuffers/protobuf/releases/download/v3.7.0/protobuf-all-3.7.0.tar.gz
$ tar xfvz protobuf-all-3.7.0.tar.gz
$ cd protobuf-3.7.0
$ ./configure
$ make
$ sudo make install
$ sudo ldconfig
```

3. Install tensorflow/models
```bash
$ git clone -b v1.13.0 https://github.com/tensorflow/models.git
$ cd models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=/home/dandelion/dev/models/research:/home/dandelion/dev/models/research/slim:/home/dandelion/dev/models/research/object_detection:${PYTHONPATH}
```

4. Clone the project repository
```bash
https://github.com/atinfinity/CarND-Capstone.git
```

5. Make and run styx
```bash
$ cd CarND-Capstone/ros
$ catkin_make
$ source devel/setup.sh
$ roslaunch launch/styx.launch
```

6. Run the simulator

### Training
- I made [training script](https://github.com/atinfinity/CarND-Capstone/blob/master/tl-detection/train.py) to detect and classify traffic light.
- I used [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
- I choiced `ssd_mobilenet_v2_coco` from [detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) as `fine_tune_checkpoint`. Because, this network model is lightweight and suitable for real time processing.

#### Dataset
I used [this dataset](https://github.com/alex-lechner/Traffic-Light-Classification#1-the-lazy-approach) to train the model of object detection.

#### Dockerfile
I make [Dockerfile](https://github.com/atinfinity/CarND-Capstone/blob/master/tl-detection/Dockerfile) to train the model of object detection.

#### Create TFRecord
```bash
$ python create_tf_record.py --data_dir=simulator_dataset_rgb/Green/,simulator_dataset_rgb/Red/,simulator_dataset_rgb/Yellow/ --label_map_path=data/udacity_label_map.pbtxt --output_path=data/train.record
```

#### Training
```bash
$ python train.py --logtostderr --train_dir=./model --pipeline_config_path=config/ssd_inception_v2_coco.config
```

#### Export frozen graph
```bash
$ python export_inference_graph.py --input_type=image_tensor --pipeline_config_path=config/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix=model/model.ckpt-200000 --output_directory=frozen_model/
```
