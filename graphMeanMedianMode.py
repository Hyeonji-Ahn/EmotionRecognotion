from collections import Counter
from statistics import median, mean
import csv
from contextlib import AsyncExitStack
import numpy as np
import cv2
import sys
import math
import glob
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import scipy.interpolate
import copy
import os
import pandas as pd
from PIL import Image
import dlib
import csv

def compute_statistics(numbers, outlier_percent=10, threshold=1.5):
    # First, we will remove all values below the threshold
    filtered_numbers = [num for num in numbers if num >= threshold]
    # Next, we'll sort the numbers
    sorted_numbers = sorted(filtered_numbers)

    # Determine the number of outliers to exclude
    k = int(len(sorted_numbers) * outlier_percent / 100)

    # Exclude the k largest numbers
    numbers_without_outliers = sorted_numbers[:-k]

    # Compute the mode using Counter
    mode_counter = Counter(numbers_without_outliers)
    mode = max(mode_counter.keys(), key=lambda x: mode_counter[x])

    # Compute the median using the statistics module
    med = median(numbers_without_outliers)

    # Compute the mean using the statistics module
    avg = mean(numbers_without_outliers)

    return mode, med, avg

FixedscaleValues = [3,0.8, 0.1] # random value here. format should be [range,max,min]

class grid:
    """The grid class is the main class of pydic. This class embed a lot of usefull
method to treat and post-treat results"""

    def __init__(self, grid_x, grid_y, size_x, size_y):
        """Construct a new grid objet with x coordinate (grid_x),
             y coordinate (grid_y), number of point along x (size_x) and
             number of point along y (size_y)"""
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.size_x = size_x
        self.size_y = size_y
        self.disp_x = self.grid_x.copy().fill(0.)
        self.disp_y = self.grid_y.copy().fill(0.)
        # try to add disp_xy
        self.disp_xy_indi = self.grid_x.copy().fill(0.)
        self.strain_xx = None
        self.strain_yy = None
        self.strain_xy = None

def detect_face_and_nose(image_path):
    # Load the face detector from dlib
    face_detector = dlib.get_frontal_face_detector()
    
    # Load the facial landmark detector from dlib
    landmark_predictor = dlib.shape_predictor(r"C:\Users\panda\OneDrive\Desktop\FacialRecTests\EmotionRecognotion-2\shape_predictor_68_face_landmarks.dat") #CHANGE FILE PATH

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector(gray)
    
    if len(faces) == 0:
        print("No faces found in the image.")
        return
    
    # Assuming only one face in the image, extract the first face
    face = faces[0]

    # Detect facial landmarks for the detected face
    landmarks = landmark_predictor(gray, face)

    # Get the center coordinates of the nose from the facial landmarks
    nose_coords = (landmarks.part(30).x, landmarks.part(30).y)

    return nose_coords

def read_dic_file(mainDir, grid_size_px, *args, **kwargs): #HEREREERERER
    # read meta info file
    meta_info = {}
    if 'meta_info_file' in kwargs:
        print('read meta info file', kwargs['meta_info_file'], '...')
        with open(kwargs['meta_info_file']) as f:
            lines = f.readlines()
            # header = lines[1] # debugging
            header = lines[0]
            field = header.split()
            for l in lines[1:-1]:
                val = l.split()
                if len(val) > 1:
                    dictionary = dict(zip(field, val))
                    meta_info[val[0]] = dictionary

    #Actual important Part

    dic_file = mainDir +'/result.dic' 
    image_pattern = mainDir+'/*.png'
    img_list = sorted(glob.glob(image_pattern))
    assert len(img_list) > 1, "there is not image in " + str(image_pattern)

    image = cv2.imread(img_list[0], 0)
    y_1, x_1 = image.shape 
    area_of_intersest = [(0, 0), (x_1, y_1)]
    area = area_of_intersest

    points = []
    points_x = np.float64(np.arange(area[0][0], area[1][0], grid_size_px[0]))
    points_y = np.float64(np.arange(area[0][1], area[1][1], grid_size_px[1]))

    for x in points_x:
        for y in points_y:
            points.append(np.array([np.float32(x), np.float32(y)]))
    points = np.array(points)

    points_in = remove_point_outside(points, area, shape='box')
    number_alpha = 0

    image = Image.open(img_list[0])
    # Check if the image has an alpha channel (RGBA or LA mode)
    if image.mode in ('RGBA', 'LA'):
        # Get the pixel data (a 2-dimensional array of (R, G, B, A) tuples)
        pixel_data = image.load()

        # Loop through all the pixels
        for j in range(len(points_in)):
                # Get the pixel value at (x, y)
                pixel = pixel_data[points_in[j][0],points_in[j][1]]

                # Check if the alpha value (A) is zero
                if len(pixel) == 4 and pixel[3] == 0:
                    number_alpha += 1
    print(number_alpha)
    result_file = r"C:\Users\panda\Desktop\KetPatients\ket-eyeblack\4a" + "/result.dic"
    # first read grid

    with open(result_file) as f:
        head = f.readlines()[0:2]
    (xmin, xmax, xnum, win_size_x) = [float(x) for x in head[0].split()]
    (ymin, ymax, ynum, win_size_y) = [float(x) for x in head[1].split()]

    # print(xmin, xmax, xnum, win_size_x)
    # print(ymin, ymax, ynum, win_size_y)

    grid_x, grid_y = np.mgrid[xmin:xmax:int(xnum)*1j, ymin:ymax:int(ynum)*1j]
    mygrid = grid(grid_x, grid_y, int(xnum), int(ynum))

    # the results
    grid_list = []
    point_list = []
    image_list = []

    # parse the result file
    with open(result_file) as f:
        res = f.readlines()[2:-1]
        for line in res: #per frame
            val = line.split('\t')
            image_list.append(val[0]) #image list
            point = []
            for pair in val[1:-1]: 
                (x, y) = [float(x) for x in pair.split(',')]
                point.append(np.array([np.float32(x), np.float32(y)]))
            point_list.append(np.array(point)) 
            grid_list.append(copy.deepcopy(mygrid))
    f.close()

    image = cv2.imread(image_list[0])
    nose_coords = detect_face_and_nose(image)
    print(nose_coords)

    if nose_coords is None:
        print("Nose not detected in the first image. Aborting.")
        exit()

    for k in range(3):
        if(k == 0):
            title = "Mode"
            #all min max stay same
        elif(k == 1):
            title = "Median"
        elif(k==2):
            title = "Mean"
        x_avgDist = []
        y_avgDist = []

        number_alpha = 0
        print(title," // ","xmin: ",xmin, " xmax: ",xmax," ymin: ",ymin," ymax: ",ymax)

        image = Image.open(img_list[0])
        # Check if the image has an alpha channel (RGBA or LA mode)
        if image.mode in ('RGBA', 'LA'):
            # Get the pixel data (a 2-dimensional array of (R, G, B, A) tuples)
            pixel_data = image.load()

            # Loop through all the pixels
            for j in range(len(points_in)):
                    # Get the pixel value at (x, y)
                    x = points_in[j][0]
                    y = points_in[j][1]
                    if x>xmin and x<xmax and y>ymin and y<ymax:
                        pixel = pixel_data[points_in[j][0],points_in[j][1]]
                        # Check if the alpha value (A) is zero
                        if len(pixel) == 4 and pixel[3] == 0:
                            number_alpha += 1
        print("alpha coordinates: ",number_alpha)

        numPoint = 0
        for i in range(1, len(point_list)-1):
            distance = []
            numPoint = 0
            curFrame = point_list[0];
            nextFrame = point_list[i];
            distSum = 0;
            xlist = []
            ylist = []
            for coor in range(len(curFrame)):
                x = nextFrame[coor][0]
                y = nextFrame[coor][1]
                if x>xmin and x<xmax and y>ymin and y<ymax:
                    numPoint+=1
                    dx = nextFrame[coor][0] - curFrame[coor][0]
                    xlist.append(dx)
                    dy = nextFrame[coor][1] - curFrame[coor][1]
                    ylist.append(dy)
            xmode, xmedian, xmean_value = compute_statistics(xlist)
            ymode, ymedian, ymean_value = compute_statistics(ylist)
            for coor in range(len(curFrame)):
                x = nextFrame[coor][0]
                y = nextFrame[coor][1]
                if x>xmin and x<xmax and y>ymin and y<ymax:
                    numPoint+=1
                    dx = nextFrame[coor][0] - curFrame[coor][0] - xmode
                    dy = nextFrame[coor][1] - curFrame[coor][1] - ymode
                    distSum = (dx ** 2 + dy **2) ** 0.5
                avgDist = distSum / (len(curFrame) - number_alpha) 
                distance.append(distSum)
            print(0,i,": ",avgDist," total points:",(numPoint-number_alpha))
            mode, median, mean_value = compute_statistics(distance)
            print("Mode:", mode) # Output will be 25
            print("Median:", median) # Output will be 25
            print("Mean:", mean_value) # Output will be 22.5
            x_avgDist.append(i)
            if k == 0:
                y_avgDist.append(mode)
            elif k == 1:
                y_avgDist.append(median)
            elif k == 2:
                y_avgDist.append(mean_value)
        plt.plot(x_avgDist, y_avgDist)
        
        # naming the x axis
        plt.xlabel('Frame')
        # naming the y axis
        plt.ylabel('Mode')
        
        # giving a title to my graph
        plt.title(title)
        
        # function to show the plot
        output_folder = mainDir + "/graph"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, title+".png"))
        plt.show()

        cv2.waitKey(0)
        
        fileName = output_folder+"/"+title+".csv"
        directory = os.path.dirname(fileName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        csv_file = open(fileName, 'w+')
        csvWriter = csv.writer(csv_file, delimiter=',')
        for i in  range(len(x_avgDist)):
            csvWriter.writerow([x_avgDist[i], y_avgDist[i]])
        print(title, " Saved!")
    plt.close()

area = []
cropping = False

def remove_point_outside(points, area,  *args, **kwargs):
    shape = 'box' if not 'shape' in kwargs else kwargs['shape']
    xmin = area[0][0]
    xmax = area[1][0]
    ymin = area[0][1]
    ymax = area[1][1]
    res = []
    for p in points:
        x = p[0]
        y = p[1]
        # if ((x == xmin) and (x == xmax) and (y == ymin) and (y == ymax)):   #debugging
        if ((x >= xmin) and (x <= xmax) and (y >= ymin) and (y <= ymax)):
            res.append(p)
    return np.array(res)

class Plot:
    def __init__(self, image, grid, data, title):
        self.data = np.ma.masked_invalid(data)
        self.data_copy = np.copy(self.data)
        self.grid_x = grid.grid_x
        self.grid_y = grid.grid_y
        self.data = np.ma.array(self.data, mask=self.data == np.nan)
        # need to add a mask file

        self.title = title
        self.image = image

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.25, bottom=0.25)

        self.ax.imshow(image, cmap=plt.cm.binary)
        #ax.contour(grid_x, grid_y, g, 10, linewidths=0.5, colors='k', alpha=0.7)

        self.im = self.ax.contourf(grid.grid_x, grid.grid_y, self.data, 10, cmap=plt.cm.rainbow,
                                   vmax=self.data.max(), vmin=self.data.min(), alpha=0.7)
        self.contour_axis = plt.gca()

        self.ax.set_title(title)
        self.cb = self.fig.colorbar(self.im)

        axmin = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        axmax = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])
        self.smin = Slider(axmin, 'set min value', self.data.min(
        ), self.data.max(), valinit=self.data.min(), valfmt='%1.6f')
        self.smax = Slider(axmax, 'set max value', self.data.min(
        ), self.data.max(), valinit=self.data.max(), valfmt='%1.6f')

        self.smax.on_changed(self.update)
        self.smin.on_changed(self.update)

    def update(self, val):
        self.contour_axis.clear()
        self.ax.imshow(self.image, cmap=plt.cm.binary)
        self.data = np.copy(self.data_copy)
        self.data = np.ma.masked_where(self.data > self.smax.val, self.data)
        self.data = np.ma.masked_where(self.data < self.smin.val, self.data)
        self.data = np.ma.masked_invalid(self.data)

        self.im = self.contour_axis.contourf(
            self.grid_x, self.grid_y, self.data, 10, cmap=plt.cm.rainbow, alpha=0.7)

        self.cb.update_bruteforce(self.im)
        self.cb.set_clim(self.smin.val, self.smax.val)
        self.cb.set_ticks(np.linspace(self.smin.val, self.smax.val, num=10))

read_dic_file(r"C:\Users\panda\Desktop\KetPatients\ket-eyeblack\4a", (10,10), interpolation='raw', save_image=True, scale_disp=1, scale_grid=1) #CHANGE FILE PATH
#def read_dic_file(mainDir, grid_size_px, *args, **kwargs):