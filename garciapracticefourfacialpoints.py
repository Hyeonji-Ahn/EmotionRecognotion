#practice emotions
import dlib
import cv2
import numpy as np
import os

def detect_face_landmarks(image_path, output_folder):
    # Load the face detector and landmark predictor models from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\ahj28\Desktop\Python\shape_predictor_68_face_landmarks.dat")  # You need to download this model file

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray_image)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Assuming there is only one face in the image
    face = faces[0]

    # Predict 68 facial landmarks for the detected face
    landmarks = predictor(gray_image, face)

    # Extract the coordinates of the required points
    left_eye_x, left_eye_y = landmarks.part(36).x, landmarks.part(37).y
    right_eye_x, right_eye_y = landmarks.part(45).x, landmarks.part(46).y
    nose_x, nose_y = landmarks.part(30).x, landmarks.part(30).y

    leftEye_high = max(landmarks.part(38).y,landmarks.part(39).y)
    rightEye_high = max(landmarks.part(44).y,landmarks.part(45).y)
    leftr = landmarks.part(20).y - leftEye_high
    rightr = landmarks.part(25).y - rightEye_high
    leftPoint = [ landmarks.part(19).x , landmarks.part(20).y + leftr//2 ] 
    rightPoint = [landmarks.part(24).x, landmarks.part(24).y + rightr//2 ] 

    # Calculate the points as specified
    left_cheek_x = left_eye_x
    left_cheek_y = nose_y

    right_cheek_x = right_eye_x
    right_cheek_y = nose_y

    # Draw the points on the image for visualization
    cv2.circle(image, (leftPoint[0], leftPoint[1]), 20, (255, 0, 0), -1)
    cv2.circle(image, (rightPoint[0], rightPoint[1]),20, (0, 0, 0), -1)
    cv2.circle(image, (left_cheek_x, left_cheek_y), 20, (0, 255, 0), -1)
    cv2.circle(image, (right_cheek_x, right_cheek_y), 20, (0, 0, 255), -1)

    # Save the image with points in the output folder
    filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_folder, "marked_" + filename)
    cv2.imwrite(output_image_path, image)

    # Return the points as an array
    return [
        [leftPoint[0], leftPoint[1]],
        [rightPoint[0], rightPoint[1]],
        [left_cheek_x, left_cheek_y],
        [right_cheek_x, right_cheek_y]
    ]

if __name__ == "__main__":
    image_path = r"C:\Users\ahj28\Desktop\Garcia DISC Data\Twins\2697\CroppedRect\002.png"  # Replace with the path to the image you want to process
    output_folder = r"C:\Users\ahj28\Desktop\Garcia DISC Data\Twins\2697"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    points = detect_face_landmarks(image_path, output_folder)
    print(points)
