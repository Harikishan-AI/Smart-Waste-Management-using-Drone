import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
import torch
import torchvision.transforms as transforms
from tensorflow.keras.applications import mobilenet_v3, vgg16
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
import random  # For simulating GPS coordinates

# Set paths to YOLO model configuration, weights, and class names file
YOLO_CONFIG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
COCO_NAMES = "coco.names"

# Load YOLO class labels from coco.names file
with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model with pre-trained weights and configuration
yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# Get YOLO output layers for making predictions
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load pre-trained image classification models
mobilenet_model = mobilenet_v3.MobileNetV3Small(weights='imagenet')
vgg_model = vgg16.VGG16(weights='imagenet')
alexnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
alexnet_model.eval()

# Function to simulate getting GPS coordinates from a drone’s GPS
def get_gps_coordinates():
    # Random GPS coordinates for simulation (e.g., latitude and longitude)
    latitude = round(28.7041 + random.uniform(-0.005, 0.005), 6)
    longitude = round(77.1025 + random.uniform(-0.005, 0.005), 6)
    return f"{latitude}, {longitude}"

# Function to preprocess image for each model
def preprocess_image(image_path, size=(224, 224), model_type="mobilenet"):
    img = Image.open(image_path).resize(size)  # Open and resize image
    img_array = np.array(img)  # Convert image to numpy array
    if model_type == "mobilenet":
        img_array = mobilenet_preprocess(img_array)  # Preprocess for MobileNet
    elif model_type == "vgg16":
        img_array = vgg16_preprocess(img_array)  # Preprocess for VGG16
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model input
    return img_array

# Function to classify image with MobileNetV3
def mobilenet_classify(image_path):
    image = preprocess_image(image_path, (224, 224), "mobilenet")  # Preprocess image
    preds = mobilenet_model.predict(image)  # Make prediction
    return mobilenet_v3.decode_predictions(preds, top=1)[0][0][1]  # Get top prediction label

# Function to classify image with VGG16
def vgg16_classify(image_path):
    image = preprocess_image(image_path, (224, 224), "vgg16")  # Preprocess image
    preds = vgg_model.predict(image)  # Make prediction
    return vgg16.decode_predictions(preds, top=1)[0][0][1]  # Get top prediction label

# Function to classify image with AlexNet (PyTorch)
def alexnet_classify(image_path):
    img = Image.open(image_path)  # Open image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Preprocess and add batch dimension
    with torch.no_grad():
        output = alexnet_model(img_tensor)  # Make prediction
    return torch.argmax(output, 1).item()  # Get top prediction class ID

# Function to perform object detection with YOLO
def yolo_detect(image_path):
    image = cv2.imread(image_path)  # Read input image
    height, width = image.shape[:2]  # Get image dimensions
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # Preprocess image for YOLO
    yolo_net.setInput(blob)  # Set blob as input to YOLO
    outputs = yolo_net.forward(output_layers)  # Get YOLO predictions

    # Initialize lists to store detection details
    boxes, confidences, class_ids = [], [], []

    # Process each output layer's predictions
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Convert detection to bounding box coordinates
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])  # Add box coordinates
                confidences.append(float(confidence))  # Add confidence score
                class_ids.append(class_id)  # Add class ID

    # Apply Non-Maximum Suppression to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = [(boxes[i], classes[class_ids[i]], confidences[i]) for i in indices.flatten()]

    # Draw bounding boxes on the image
    for box, label, conf in detections:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display result image
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections  # Return detection results

# Function to send email notification to authority
def send_email(detections, location):
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "your_email@gmail.com"  # Sender's email
    receiver_email = "authority_email@example.com"  # Receiver's email
    password = "your_password"  # Password for sender's email

    # Setup email message content
    message = MIMEMultipart("alternative")
    message["Subject"] = "Detected Object Alert"
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create detection details for the email body
    detection_info = "\n".join([f"{label} with {confidence*100:.2f}% confidence" for _, label, confidence in detections])
    body = f"""
    The following objects were detected:
    {detection_info}
    
    Location coordinates: {location}
    """
    message.attach(MIMEText(body, "plain"))  # Attach message body

    # Send the email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Enable TLS
        server.login(sender_email, password)  # Login to SMTP server
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()  # Close the server

# Main function to run the pipeline
def main(image_path):
    # Get GPS coordinates from drone
    location_coords = get_gps_coordinates()
    
    # Image classification
    mobile_class = mobilenet_classify(image_path)
    vgg_class = vgg16_classify(image_path)
    alex_class = alexnet_classify(image_path)
    print("MobileNetV3 Prediction:", mobile_class)
    print("VGG16 Prediction:", vgg_class)
    print("AlexNet Prediction:", alex_class)

    # Object detection with YOLO
    detections = yolo_detect(image_path)
    print("YOLO Detections:", detections)

    # Send email to authority with detection details and GPS coordinates
    send_email(detections, location_coords)

# Example usage:
image_path = "path_to_image.jpg"  # Replace with your image file path
main(image_path)
