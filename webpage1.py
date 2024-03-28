import streamlit as st
from PIL import Image
import numpy as np
import cv2



def detect_skin(image):
    # Convert image to numpy array
    img_array = np.array(image)
    # Convert color space from RGB to YCrCb
    img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    # Define lower and upper bounds for skin color in YCrCb space
    lower_skin = np.array([0, 133, 77], dtype="uint8")
    upper_skin = np.array([255, 173, 127], dtype="uint8")
    # Create a mask to extract skin pixels
    mask_skin = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    # Apply the mask to the original image
    result = cv2.bitwise_and(img_array, img_array, mask=mask_skin)
    # Convert result back to PIL Image
    result_image = Image.fromarray(result)
    return result_image

def extract_skin_color(image):
    # Convert image to numpy array
    img = np.array(image)
    
    # Convert RGB to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    
    # Apply the mask to the image
    skin = cv2.bitwise_and(img, img, mask=mask)
    
    # Get the average RGB color of the skin area
    avg_skin_color = np.mean(skin, axis=(0, 1)).astype(int)
    
    return avg_skin_color

def main():
    st.title("Skin Tone Recognition App")
    st.write(
        """
        Welcome to the Skin Tone Recognition App!

        This app detects skin tone in the provided image and displays the result.
        """
    )

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect skin tone in the image
        skin_image = detect_skin(image)
        st.image(skin_image, caption="Detected Skin Tone", use_column_width=True)
        skin_color = extract_skin_color(image)

        st.write(f"The average skin color (RGB) in the image is: {skin_color}")


#Example of predicting the seasonal color palette for a new skin tone sample
new_sample = [[20, 50, 200]]  # RGB values for the new skin tone 
#in this case i want to add user image 
predicted_palette = clf.predict(extract_skin_color)


print("Predicted seasonal color palette:", predicted_palette)
if __name__ == "__main__":
    main()
