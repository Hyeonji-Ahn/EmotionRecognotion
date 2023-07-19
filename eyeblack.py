#Eye Black


# Directory containing the images
inDir_black = mainDir + "\Cropped"
outDir_black = mainDir+"\EyeBlacked"

if not os.path.exists(outDir_black):
    os.makedirs(outDir_black)

def black_eyes(img):
   eyes=[]
   # convert to grayscale of each frames
   gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # read the haarcascade to detect the faces in an image
   eye_cascade = cv2.CascadeClassifier(r"C:\Users\ahj28\Desktop\Python\haarcascade_eye_tree_eyeglasses.xml")

   # detects faces in the input image
      # detects eyes of within the detected face area (roi)
   eyes.extend(eye_cascade.detectMultiScale(gray_frame))
   print(eyes)
      # draw a rectangle around eyes
   return eyes
      #for (ex,ey,ew,eh) in eyes:

def crop_images(images):
    cropped_images = []
    i = 0
    # Crop each image based on the box coordinates
    for image in images:
        i +=1
# taking half of the width:
        for i in range(eyes[0][0], eyes[0][0]+eyes[0][2]):
            for j in range(eyes[0][1], eyes[0][1]+eyes[0][3]):
                image[j, i] = (1,1,1)
        for i in range(eyes[1][0], eyes[1][0]+eyes[1][2]):
            for j in range(eyes[1][1], eyes[1][1]+eyes[1][3]):
                image[j, i] = (1,1,1)
        cropped_images.append(image)
    return cropped_images

# Get a list of image file names in the directory
image_files = [f for f in os.listdir(inDir_black) if f.endswith('.png')]



# Load the images
images = [cv2.imread(os.path.join(inDir_black, image_file)) for image_file in image_files]


# Crop the series of images
eyes = black_eyes(images[0])
cropped_images = crop_images(images)
#cropped_images = crop_images(images)

def returnZero(currentI, desiredNum):
    ans = "";
    for i in range (desiredNum - len(str(currentI))):
        ans += "0"
    return ans

#Display and save the cropped images
for i, cropped_image in enumerate(cropped_images):
    cv2.imwrite(os.path.join(outDir_black, returnZero(i+1,len(str(len(cropped_images)))) + f"{i+1}.png"), cropped_image)
