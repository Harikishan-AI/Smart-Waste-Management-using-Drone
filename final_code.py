import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
import random  # For simulating GPS coordinates

# Load pre-trained MobileNetV3 model
mobilenet_model = mobilenet_v3.MobileNetV3Small(weights='imagenet')

# Function to simulate getting GPS coordinates from a drone’s GPS
def get_gps_coordinates():
    # Random GPS coordinates for simulation (e.g., latitude and longitude)
    latitude = round(28.7041 + random.uniform(-0.005, 0.005), 6)
    longitude = round(77.1025 + random.uniform(-0.005, 0.005), 6)
    return f"{latitude}, {longitude}"

# Function to preprocess the image for MobileNetV3
def preprocess_image(image_path, size=(224, 224)):
    img = Image.open(image_path).resize(size)  # Open and resize image
    img_array = np.array(img)  # Convert image to numpy array
    img_array = mobilenet_preprocess(img_array)  # Preprocess for MobileNetV3
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model input
    return img_array

# Function to classify the image with MobileNetV3
def mobilenet_classify(image_path):
    image = preprocess_image(image_path, (224, 224))  # Preprocess image
    preds = mobilenet_model.predict(image)  # Make prediction
    # Decode predictions to get the top label and confidence
    label, confidence = mobilenet_v3.decode_predictions(preds, top=1)[0][0][1], mobilenet_v3.decode_predictions(preds, top=1)[0][0][2]
    return label, confidence

# Function to send email notification with classification result and location
def send_email(classification_result, location):
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "your_email@gmail.com"  # Sender's email
    receiver_email = "authority_email@example.com"  # Receiver's email
    password = "your_password"  # Password for sender's email

    # Setup email message content
    message = MIMEMultipart("alternative")
    message["Subject"] = "Classification Alert: MobileNetV3 Result"
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create email body with classification result and location
    body = f"""
    The object detected has been classified as:
    {classification_result[0]} with {classification_result[1]*100:.2f}% confidence
    
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

    # Image classification with MobileNetV3
    label, confidence = mobilenet_classify(image_path)
    print("MobileNetV3 Classification Result:", label, confidence)

    # Send email to authority with classification result and GPS coordinates
    send_email((label, confidence), location_coords)

# Example usage:
image_path = "path_to_image.jpg"  # Replace with your image file path
main(image_path)
