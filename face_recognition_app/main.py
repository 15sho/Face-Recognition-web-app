import face_recognition
import cv2
import os


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



while True:
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

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
