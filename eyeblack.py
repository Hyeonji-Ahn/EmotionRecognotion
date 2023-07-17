# import required libraries
import cv2
import numpy as np
import os
from PIL import Image
def black_eyes(img):
   eyes=[]
   # convert to grayscale of each frames
   gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # read the haarcascade to detect the faces in an image
   face_cascade = cv2.CascadeClassifier(r"C:\Users\panda\Downloads\eyetracker\haarcascade_frontalface_default.xml")
   eye_cascade = cv2.CascadeClassifier(r"C:\Users\panda\Downloads\eyetracker\haarcascade_eye_tree_eyeglasses.xml")

   # detects faces in the input image
   faces = face_cascade.detectMultiScale(gray_frame, 1.3, 4)
   print('Number of detected faces:', len(faces))
   # loop over the detected faces
   for (x,y,w,h) in faces:
      roi_gray = gray_frame[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      
      # detects eyes of within the detected face area (roi)
      eyes.extend(eye_cascade.detectMultiScale(roi_gray))
      # draw a rectangle around eyes
   return eyes, roi_color
      #for (ex,ey,ew,eh) in eyes:

# def crop_images(images, eyes, roi_color):
#    cropped_images = []
#    i = 0
#    # Crop each image based on the box coordinates
#    for image in images:
#       i +=1
  
# # rectangle box yay 
#    for (ex,ey,ew,eh) in eyes:
#       cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
#    cropped_images.append(image)

#    return cropped_images

# # Directory containing the images
#  # Replace with the actual directory path
# #outDir = r"C:\Users\ahj28\Desktop\Python\SampleOutput430sec\Cropped"
# inDir = r"C:\Users\panda\Downloads\ShirleyFPS"
# outDir = r"C:\Users\panda\OneDrive\Desktop\FacialRecTests\7.14.23\FinalImages"

# if not os.path.exists(outDir):
#     os.makedirs(outDir)

# # Get a list of image file names in the directory
# image_files = [f for f in os.listdir(inDir) if f.endswith('.png')]

# # Load the images
# images = [cv2.imread(os.path.join(inDir, image_file)) for image_file in image_files]


# # Crop the series of images
# eyes, roi_color = black_eyes(images[0])
# cropped_images = crop_images(images, eyes, roi_color)

# def returnZero(currentI, desiredNum):
#     ans = "";
#     for i in range (desiredNum - len(str(currentI))):
#         ans += "0"
#     return ans

# # Display and save the cropped images
# for i, cropped_image in enumerate(cropped_images):
#     cv2.imwrite(os.path.join(outDir, returnZero(i+1,len(str(len(cropped_images)))) + f"{i+1}.png"), cropped_image)
