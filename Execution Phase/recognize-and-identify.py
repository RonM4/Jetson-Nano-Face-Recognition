# Step 1: Loading trained model.
# Step 2: Accessing Camera Stream.
# Step 3: Face detection via FaceDetect NVIDIA model.
# Step 4: Face recognition via trained model.
# Note: Import Jetson lib. before the rest
# If not - the glDisplay window won't open

import jetson_inference
import jetson_utils
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import time
import collections

#########################################
# ** Cameras and Display Config      ** #
#########################################

camera = jetson_utils.videoSource("/dev/video0")
display = jetson_utils.videoOutput()

#########################################
# ** Model functions and definitions ** #
#########################################

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('training-results', transform=transform)

# Load the model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

model.load_state_dict(torch.load('face_recognition_model.pth'))

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Preprocessing the image
def preprocess_image(image, transform):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image


# Function to predict the person in the image
def predict(image, model, transform, device, threshold=0.7):
    image_tensor = preprocess_image(image, transform)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = nn.Softmax(dim=1)(outputs)
        max_prob, preds = torch.max(probabilities, 1)

        if max_prob.item() < threshold:
            return "Unknown"

    return dataset.classes[preds[0]]

#########################################
#           ** Main Loop **             #
#########################################	

print("Loading detection model...")
faceDetector = jetson_inference.detectNet("facedetect", threshold = 0.8)
print("Done.")

# Initializing values
recognized_faces = collections.deque(maxlen=5)
pred_recognized_person = "Unknown"
predicted_person_label = "Unknown"

while True: #display.IsStreaming():
	frame = camera.Capture()
	face_detections = faceDetector.Detect(frame, overlay = "none")

	if frame is None:
		continue

	np_img = jetson_utils.cudaToNumpy(frame)
	cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)

	for i, detection in enumerate(face_detections):

		# Aquiring Bounding Box Coordinates
		# Doc: https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html
		left = int(detection.Left)
		top = int(detection.Top)
		right = int(detection.Right)
		bottom = int(detection.Bottom)
		
		# Cropping the face from the image
		extracted_face = cv_img[top:bottom, left:right]
		
		# Passing extracted face image to trained model
		model_threshold = 0.9
		pred_recognized_person = predict(extracted_face, model, transform, device, model_threshold)
		
		recognized_faces.append(pred_recognized_person)
		
		# Update the label only if was recognized in the last 5 frames
		if recognized_faces.count(pred_recognized_person) >= 3:
			predicted_person_label = pred_recognized_person		
		
		# Annotating image	
		cv2.rectangle(cv_img, (left, top), (right, bottom), (46, 139, 87), 2)

		cv2.putText(cv_img, predicted_person_label, (left, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (46, 139, 87), 2)
		print("Identified: ", predicted_person_label)

	# Preparing image for rendering and display    
	bgr_img = jetson_utils.cudaFromNumpy(cv_img, isBGR=True)
	rgb_img = jetson_utils.cudaAllocMapped(width=bgr_img.width, height=bgr_img.height, format='rgb8')
	jetson_utils.cudaConvertColor(bgr_img, rgb_img)
	    
	# Displaying image
	display.Render(rgb_img)		

	# Updating Status Bar
	display.SetStatus("Facenet {:.0f} FPS | # faces :{:d}".format(faceDetector.GetNetworkFPS(), len(face_detections)))
