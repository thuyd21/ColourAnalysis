import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import numpy as np
import cv2


## PHASE 1 
# Load the dataset
data = pd.read_csv('models.csv')

# Load the dataset
data1 = pd.read_csv('faces.csv')

# Drop the "date" column
data1.drop("date", axis=1, inplace=True)

#combine two dataset 

df = pd.concat([data, data1], axis= 0)

# Drop the "n_covers" column
df.drop("n_covers", axis=1, inplace=True)

## PHASE 2
# Convert dataset to DataFrame
df1 = pd.DataFrame(df)

# Function to convert hex color code to RGB
def hex_to_rgb(tone):
    tone = tone.lstrip('#')
    return tuple(int(tone[i:i+2], 16) for i in (0, 2, 4))

# Apply function to convert hex color codes to RGB
df1['RGB'] = df1['tone'].apply(hex_to_rgb)

# rename to make it clearer for viewers 

df1.rename(columns={'tone': 'hexcode'}, inplace=True)
df1.rename(columns={'l': 'lightness'}, inplace=True)



## PHASE 3 - COLOUR CLASSIFICATION
#RBG CHOSEN ATTEMPT
# Extract RGB components from the 'RGB' column
df1['red'] = df1['RGB'].apply(lambda x: x[0])  # Extract red component
df1['green'] = df1['RGB'].apply(lambda x: x[1])  # Extract green component
df1['blue'] = df1['RGB'].apply(lambda x: x[2])  # Extract blue component

# Define threshold ranges for each season based on RGB values
spring_ranges_rgb = {
    'red': (210, 250),     # High red
    'green': (150, 217),   # Moderate green
    'blue': (148, 205)     # Moderate blue
}

summer_ranges_rgb = {
    'red': (100, 200),   # Moderate red
    'green': (0, 250),   # High green
    'blue': (0, 250)     # High blue
}

autumn_ranges_rgb = {
    'red': (210, 255),     # High red
    'green': (120, 190),   # Moderate green
    'blue': (0, 150)       # Low to moderate blue
}

winter_ranges_rgb = {
    'red': (0, 250),     # Low to moderate red
    'green': (0, 180),   # Moderate green
    'blue': (0, 255)     # High blue
}

# Function to classify individuals into seasonal color palettes based on RGB values
def classify_season_rgb(row):
    red, green, blue = row['red'], row['green'], row['blue']
    
    if spring_ranges_rgb['red'][0] <= red <= spring_ranges_rgb['red'][1] \
        and spring_ranges_rgb['green'][0] <= green <= spring_ranges_rgb['green'][1] \
        and spring_ranges_rgb['blue'][0] <= blue <= spring_ranges_rgb['blue'][1]:
        return 'Spring'
    
    elif summer_ranges_rgb['red'][0] <= red <= summer_ranges_rgb['red'][1] \
        and summer_ranges_rgb['green'][0] <= green <= summer_ranges_rgb['green'][1] \
        and summer_ranges_rgb['blue'][0] <= blue <= summer_ranges_rgb['blue'][1]:
        return 'Summer'
    
    elif autumn_ranges_rgb['red'][0] <= red <= autumn_ranges_rgb['red'][1] \
        and autumn_ranges_rgb['green'][0] <= green <= autumn_ranges_rgb['green'][1] \
        and autumn_ranges_rgb['blue'][0] <= blue <= autumn_ranges_rgb['blue'][1]:
        return 'Autumn'
    
    elif winter_ranges_rgb['red'][0] <= red <= winter_ranges_rgb['red'][1] \
        and winter_ranges_rgb['green'][0] <= green <= winter_ranges_rgb['green'][1] \
        and winter_ranges_rgb['blue'][0] <= blue <= winter_ranges_rgb['blue'][1]:
        return 'Winter'
    
    else:
        return 'Unknown'

# Apply classification function to each row
df1['season_rgb'] = df1.apply(classify_season_rgb, axis=1)

# Print the counts of individuals in each seasonal color palette based on RGB values
print(df1['season_rgb'].value_counts())


## PHASE 4 - MACHINE LEARNING  

# Assume 'X' contains HSL values and 'y' contains seasonal color palette labels
# Split the dataset into training and testing sets
# Split the data into features and target variable
X = df1[['red', 'green', 'blue']]
y = df1['season_rgb'] #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=385)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=385)     # change this around 

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

## PHASE 5 
# streamlit

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
    
    return avg_skin_color[:3]

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

#FINAL PHASE PREDICTING 
#predicting the seasonal color palette for a new skin tone sample
        new_sample = [extract_skin_color(image)]  # Get the average skin color from the uploaded image
        new_sample = np.array(new_sample).reshape(1, -1)  # Reshape to 2D array
        predicted_palette = clf.predict(new_sample)
        st.write("Predicted seasonal color palette:", predicted_palette)

if __name__ == "__main__":
    main()


