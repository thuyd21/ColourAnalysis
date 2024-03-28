import cv2
import numpy as np

import streamlit as st

### streamlit application to input a picture in trial 3 


 #** change hex range 
def predict_seasonal_color(hex_code):
    # Define color ranges for different seasons
    spring_colors = ["#ffcc99", "#66ccff", "#cc99ff", "#99ff99", "#ffff99"]
    summer_colors = ["#ff9999", "#99ccff", "#cc66ff", "#99ffcc", "#ffff66"]
    autumn_colors = ["#ff9966", "#99ffff", "#9966ff", "#ccff99", "#ffffcc"]
    winter_colors = ["#ff6666", "#66ffcc", "#ff66ff", "#ccffff", "#ccccff"]

    # Convert hex code to uppercase for consistency
    hex_code = hex_code.upper()

    # Calculate the color distance between the input hex code and the seasonal color palettes
    def color_distance(color1, color2):
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:], 16)
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

    spring_distance = min([color_distance(hex_code, color) for color in spring_colors])
    summer_distance = min([color_distance(hex_code, color) for color in summer_colors])
    autumn_distance = min([color_distance(hex_code, color) for color in autumn_colors])
    winter_distance = min([color_distance(hex_code, color) for color in winter_colors])

    # Determine the closest seasonal color palette

    #** double check 
    min_distance = min(spring_distance, summer_distance, autumn_distance, winter_distance)
    if min_distance == spring_distance:
        return "Spring"
    elif min_distance == summer_distance:
        return "Summer"
    elif min_distance == autumn_distance:
        return "Autumn"
    else:
        return "Winter"

#change to take an image instead of a path (st)
def extract_skin_color(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image file.")
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Extract the skin tone color from the detected face region
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = image_rgb[y:y+h, x:x+w]
        average_color = np.mean(face_region, axis=(0, 1))
        skin_color = '#{:02X}{:02X}{:02X}'.format(int(average_color[0]), int(average_color[1]), int(average_color[2]))
        return skin_color
    else:
        print("Error: No face detected in the image.")
        return None

# Example usage
image_path = input("Enter the path to your face image: ")
skin_color_hex = extract_skin_color(image_path)
if skin_color_hex:
    seasonal_color = predict_seasonal_color(skin_color_hex)
    print("Your seasonal personal color palette is:", seasonal_color)
