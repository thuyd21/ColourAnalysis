#libraries 
#streamlit is for building web application 

#http://localhost:8501
#how to import data https://www.youtube.com/watch?v=QetpwPnEpgA&ab_channel=M%C4%B1sraTurp



#Allow users to take a picture too and add comment for inputting pictures in natural lighting for the best results
import streamlit as st 
from PIL import Image 
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
def main():

# Define a VideoTransformer class to process the video frames
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
        # You can process the frame here if needed
           return frame

def main():
     st.write(":)")
st.title("Color Analysis App")
st.write(
"""
Welcome!!!

This is the Personal Seasonal Colour Analysis App.

Please follow the intructions below. To upload or capture a photo for color analysis:

:)
"""
)

name = st.text_input("Name:")
file = st.file_uploader("Select a File")

#import file of user face 
# Use webrtc_streamer to capture video from the user's camera
webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformerBase,
       # mode= input,
        async_transform=True  # Needed for async processing of frames
    )

# Add a button to take a photo
if st.button("Take Photo"):
        if webrtc_ctx is not None:
            if webrtc_ctx.video_stream:
                frame = webrtc_ctx.video_stream.get()
                if frame is not None:
                    image = Image.fromarray(frame.to_ndarray(format="bgr24"))
                    st.write("Photo Captured!")
                    st.image(image, caption="Captured Photo", use_column_width=True)

if __name__ == "__main__":
    main()



