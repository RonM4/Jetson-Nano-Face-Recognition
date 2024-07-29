## Overview

Our project uses deep learning techniques on the NVIDIA Jetson Nano platform to implement real-time face detection, alignment, and recognition.
By using the dusty-nv/jetson-inference [Docker container](https://hub.docker.com/r/dustynv/jetson-inference) for L4T-32.7.4 Nvidia JetPack 4.6.4 on a Jetson Nano 2GB, we integrate a FaceNet-based software to extract facial features (Check out [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)) on GitHub.
These features are then utilized in a custom ResNet-18 network to recognize faces accurately.
The system is designed to handle datasets with 10-15 images per person, providing robust performance even with limited data.
This approach ensures efficient and scalable facial recognition that is suitable for various applications. 

More specifically speaking, the face detection model is found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet), and the description is found [here](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/facedetectnet.html), as well as, additional info [here](https://github.com/katjasrz/FaceDetect_TRTIS).

## Structure of the project's code files

Our project contains two folders: The ```Training Phase```, and ```Execution Phase```. 

### [_Training Phase_](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/tree/main/Training%20Phase)
This folder contains two files:
* [face_extractor.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/blob/main/Training%20Phase/face-extractor.py) - This code piece is used upon the photos of people you would like the system to identify and put a nametag on the face in the frame. Please review the ```input``` and ```save-dir``` options in the parser initialization part.

After extracting the faces from the images, and saving them in a separate directory, you'll need to sort them into appropriate folders. Each person should get his \ her folder with their respectable name serving as the folder's label. The names must be accurate, as the model will be trained to use those names when tagging the faces in the frames.

* [Training_Detection.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/blob/main/Training%20Phase/Training_Detection.py) - This code piece is resposible for training the Resnet-18 model to recognize the faces in your frames. Notice the ```dataset_path``` and ```model_save_path``` variables in lines 8-9.

### Execution Phase
This folder contains a single file:

* [recognize-and-identify.py](https://github.com/4uSpock/Jetson-Nano-Face-Recognition/tree/main/Execution%20Phase) - This code piece will open a glDisplay window, where it will display the tagged, and framed faces. While integrating NVIDIA's FaceDetect model, and our model, we can frame, and nametag individuals based on their faces.