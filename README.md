## Overview

Our project uses deep learning techniques on the NVIDIA Jetson Nano platform to implement real-time face detection, alignment, and recognition.
By using the dusty-nv/jetson-inference [Docker container](https://hub.docker.com/r/dustynv/jetson-inference) for L4T-32.7.4 Nvidia JetPack 4.6.4 on a Jetson Nano 2GB, we integrate a FaceNet-based software to extract facial features (Check out [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)) on GitHub.
These features are then utilized in a custom ResNet-18 network to recognize faces accurately.
The system is designed to handle datasets with 10-15 images per person, providing robust performance even with limited data.
This approach ensures efficient and scalable facial recognition that is suitable for various applications. 

More information on the face detection model (FaceDetect) is found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet), and the description is found [here](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/facedetectnet.html), as well as, additional info [here](https://github.com/katjasrz/FaceDetect_TRTIS).

The model was trained on Python v3.10, PyTorch v1.9.1+cpu, and executed with Python 3.6.9, PyTorch 1.10.0 (as per [the container's](https://hub.docker.com/r/dustynv/jetson-inference) preconfigured version)

To install the relevant jetson-inference container, please refer to this [GitHub page](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md). The repository includes the ```run.sh``` script that will install the container automatically.


## Structure of the project's code files

Our project contains two folders: The ```Training Phase```, and ```Execution Phase```. 

### [_Training Phase_](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/tree/main/Training%20Phase)
This folder contains two files:
* [face_extractor.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/blob/main/Training%20Phase/face-extractor.py) - This code piece processes the photos of the people used for training and extracts the faces only. Please note the ```input``` and ```save-dir``` options in the parser initialization part. The code uses the coordinates of the face bounding box (as found by the FaceDetect model), and saves the extracted face found within the bounding box.

After extracting the faces from the images, and saving them in a separate directory, you'll need to sort them into appropriate folders. Each person should get his \ her folder with their respectable name serving as the folder's label. The names must be accurate, as the model will be trained to use those names when tagging the faces in the frames.

* [Training_Detection.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/blob/main/Training%20Phase/Training_Detection.py) - This code piece is resposible for training the Resnet-18 model to recognize the faces in your frames. The training contains several features:
* command-line argument parsing
*  Multiple transformations, normalization, and layer freezing to overcome the relatively small dataset.
*  Save checkpoint function to save the progress of the model at the end of each epoch (model parameters, optimizer state and current epoch).
*  StepLR Schedular - to help with loss convergence.

### [Execution Phase](https://github.com/RonM4/Jetson-Nano-Face-Recognition/tree/main/Execution%20Phase)
This folder contains a single file:

* [recognize-and-identify.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/tree/main/Execution%20Phase) - This code piece will open a glDisplay window, where it will display the tagged, and framed faces. While integrating NVIDIA's FaceDetect model, and our model, we can frame, and nametag individuals based on their faces, be it a real-time stream or a photo.
