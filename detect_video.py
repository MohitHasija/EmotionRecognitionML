import os, sys
import cv2

dir_name = os.path.dirname(os.path.abspath(__file__))
facial_recognition_library = os.path.join(dir_name, "FacialExpressionRecognition")
transforms_folder = os.path.join(facial_recognition_library, "transforms")
sys.path = ["", dir_name, facial_recognition_library, transforms_folder] + sys.path

from FacialExpressionRecognition.visualize import get_expression


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
success = False
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if not len(faces):
        print("No face detected")

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Call the pytorch function to detect emotion around face.
        img = frame[y:y + h, x:x + w]
        cv2.imwrite("image.jpg", img)
        print(get_expression(os.path.join(dir_name, "image.jpg")))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# When everything is done, release the capture
os.remove("image.jpg")
video_capture.release()
cv2.destroyAllWindows()
