#Face recognition 

#import library
import cv2 
#import module, which provides an object-oriented interface for working with filesystem paths.
import pathlib 

#creates path to xml file, which has cascade classifier which decets frontal faces
# using pathlib to create a path from cv2 module
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml" 
print(cascade_path)


clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0) #ive only got 1 camera so default 0, LIVE Camera 

#camera = cv2.VideoCapture("") #import a file and manually put it in

#imgname = "examples/obama.png"
#img = cv2.imread(imgname)


#Starts an infinite loop to continuously capture frames from the camera and process them for face detection.
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )       

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
# physical frame for the detection
    cv2.imshow("Face Detection", frame)  
    if cv2.waitKey(1) == ord("q"): # click camera screen and press q to quit
        break



camera.release()
cv2.destroyAllWindows()
