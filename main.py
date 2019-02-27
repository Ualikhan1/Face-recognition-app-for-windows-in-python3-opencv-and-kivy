import cv2
import os
import numpy as np
from PIL import Image

from kivy.app import App
from kivy.uix.button import Button

from kivy.config import Config
Config.set('graphics', 'resizable', '0');
Config.set('graphics', 'width', '640');
Config.set('graphics', 'height', '480');

class MyApp(App):
    
    def build(self):
        
        def assure_path_exists(path):
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)


        vid_cam = cv2.VideoCapture(0)

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # For each person, one face id
        face_id = 1

        # Initialize sample face image
        count = 0

        assure_path_exists("dataset/")

        # Start looping
        while(True):
            # Capture video frame
            _, image_frame = vid_cam.read()
            # Convert frame to grayscale
            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            # Detect frames of different sizes, list of faces rectangles
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            # Loops for each faces
            for (x,y,w,h) in faces:
                # Crop the image frame into rectangle
                cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
                # Increment sample face image
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                # Display the video frame, with bounded rectangle on the person's face
                cv2.imshow('frame', image_frame)
            # To stop taking video, press 'q' for at least 100ms
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            # If image taken reach 100, stop taking video
            elif count>30:
                break
        # Stop video
        vid_cam.release()
        # Close all started windows
        cv2.destroyAllWindows()
        


        # Create Local Binary Patterns Histograms for face recognization
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Using prebuilt frontal face training model, for face detection
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

        # Create method to get the images and label data
        def getImagesAndLabels(path):
            # Get all file path
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            # Initialize empty face sample
            faceSamples=[]
            # Initialize empty id
            ids = []
            # Loop all the file path
            for imagePath in imagePaths:
                # Get the image and convert it to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                # PIL image to numpy array
                img_numpy = np.array(PIL_img,'uint8')
                # Get the image id
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                # Get the face from the training images
                faces = detector.detectMultiScale(img_numpy)
                # Loop for each face, append to their respective ID
                for (x,y,w,h) in faces:
                    # Add the image to face samples
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    # Add the ID to IDs
                    ids.append(id)
            # Pass the face array and IDs array
            return faceSamples,ids
        # Get the faces and IDs
        faces,ids = getImagesAndLabels('dataset')
        # Train the model using the faces and IDs
        recognizer.train(faces, np.array(ids))
        # Save the model into trainer.yml
        assure_path_exists('trainer/')
        recognizer.save('trainer/trainer.yml')



        recognizer = cv2.face.LBPHFaceRecognizer_create()
        assure_path_exists("trainer/")
        # Load the trained mod
        recognizer.read('trainer/trainer.yml')
        # Load prebuilt model for Frontal Face
        cascadePath = "haarcascade_frontalface_default.xml"
        # Create classifier from prebuilt model
        faceCascade = cv2.CascadeClassifier(cascadePath);
        # Set the font style
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Initialize and start the video frame capture
        cam = cv2.VideoCapture(0)
        # Loop
        while True:
            # Read the video frame
            ret, im =cam.read()
            # Convert the captured frame into grayscale
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            # Get all face from the video frame
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            # For each face in faces
            for(x,y,w,h) in faces:
                # Create rectangle around the face
                cv2.rectangle(im, (x-5,y-5), (x+w+5,y+h+5), (255,0,0), 2)
                # Recognize the face belongs to which ID
                Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check the ID if exist
                if(Id == 1):
                    Id = "Ualikhan {0:.2f}%".format(round(100 - confidence, 2))
                # Put text describe who is in the picture
                cv2.rectangle(im, (x-11,y-45), (x+w+11, y-11), (255,0,0), -1)
                cv2.putText(im, str(Id), (x,y-20), font, 1, (255,255,255), 2)
            # Display the video frame with the bounded rectangle
            cv2.imshow('im',im)
            # If 'q' is pressed, close program
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # Stop the camera
        cam.release()
        # Close all windows
        cv2.destroyAllWindows()
       

if __name__ == "__main__":
    MyApp().run()
