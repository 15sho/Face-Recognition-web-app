from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
app=Flask(__name__)


# Directory where your images are stored
image_directory = 'face_recognition_app\images'  # Replace with the path to your image directory

# Load the images and their encodings
known_faces = []

# Get a reference to your webcam or video source
video_capture = cv2.VideoCapture(0)  # You can adjust the camera source as needed

# Create a dictionary to map face encodings to names
face_encodings_to_names = {}

# Iterate through the image files in the directory
for image_file in os.listdir(image_directory):
    if image_file.endswith('.jpeg'):
        image_path = os.path.join(image_directory, image_file)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]

        # Get the name from the filename (without the file extension)
        name = os.path.splitext(os.path.basename(image_path))[0]
        known_faces.append(face_encoding)

        # Store the encoding and name in the dictionary
        face_encodings_to_names[tuple(face_encoding)] = name



def gen_frames():  
    while True:
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        else:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # Compare the face encoding to known faces
                name = "Unknown"

                for known_face_encoding in known_faces:
                    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
                    if matches[0]:
                        name = face_encodings_to_names.get(tuple(known_face_encoding), "Unknown")
                        break

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with the name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)