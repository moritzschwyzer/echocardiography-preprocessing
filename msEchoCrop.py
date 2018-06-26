#
# INSTRUCTIONS:
# Define the corners with the left mouse button accordingly:
# 1st: Top center corner of the echocardiography
# 2nd: Lower left corner of the echocardiography
# 3rd: Lower right corner of the echocardiography
#
# press "c" to crop
# press "s" to save
# press "r" to reset
# press "q" to quit
#
# folder structure:
# <class>/<patientnumber>/<sequence-name(.avi/.mp4)>
# cropped/
#
# <class> needs to be provided as the argument "--folder"
#


# import the necessary packages
import argparse
import glob
import os

import cv2
import math
import numpy as np


# initialize global variables
pointMemory = []
radius = 0
cropped = False
mask = np.ones((1,1))
drawings = np.zeros((1,1))


def playvideo (cap, new_filename):
	global pointMemory
	global drawings
	global mask
	global drawings

	refPt = []
	cropping = False

	pointMemory = []
	radius = 0
	cropped = False
	ret0, frame0 = cap.read()
	mask = np.ones(frame0.shape).astype(frame0.dtype)*255
	drawings = np.zeros(frame0.shape).astype(frame0.dtype)


	cv2.setMouseCallback("video", click_and_crop)

	while(cap.isOpened()):
	    ret, frame = cap.read()

	    if ret:
	    	if cropped == False:
	    		cv2.imshow('video',frameoverlay(frame, drawings))
	    	else:
	    		cv2.imshow('video',framecrop(frame, mask))

	    else:
	    	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	    key = cv2.waitKey(1) & 0xFF
	    if key == ord('q'):
	    	return True
	    elif key == ord('s'):
	    	savevideo (cap, new_filename)
	    	break
	    elif key == ord('c'):
	    	if len(pointMemory) > 2:
	    		cropped = True if cropped == False else False
	    elif key == ord('r'):
	    	cropped = False
	    	pointMemory = []
	    	drawings = np.zeros(frame0.shape).astype(frame0.dtype)
	    	mask = np.ones(frame0.shape).astype(frame0.dtype)*255

	cap.release()

	return False


def savevideo (cap, new_filename):
	i = 0

	while(cap.isOpened()):
		
		ret, frame = cap.read()
		if i == 0:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		i += 1
		
		if ret:
			cv2.imwrite("cropped/" + new_filename + "-" + str(i).zfill(3) + ".jpg", framecrop(frame, mask))
			print("saved")

		else:
			break

 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image, mask, radius

	if event == cv2.EVENT_LBUTTONDOWN:
		pointMemory.append((x,y))
		refPt = [(x, y)]
		
		if len(pointMemory) > 3:
			return

		cropping = True
		if len(pointMemory) > 1:
			cv2.line(drawings, pointMemory[0], refPt[0], (0, 255, 0))
			if len(pointMemory) > 2:
				height, width, cchannels = drawings.shape
				stencil1 = np.zeros(drawings.shape).astype(drawings.dtype)
				stencil2 = np.zeros(drawings.shape).astype(drawings.dtype)

				radius1 = math.sqrt( ((pointMemory[0][0]-pointMemory[1][0])**2)+((pointMemory[0][1]-pointMemory[1][1])**2) )
				radius2 = math.sqrt( ((pointMemory[0][0]-pointMemory[2][0])**2)+((pointMemory[0][1]-pointMemory[2][1])**2) )
				radius = radius1 if radius1 < radius2 else radius2
				cv2.circle(stencil1, pointMemory[0], int(radius), (255, 255, 255), -1000)

				contours = [np.array([pointMemory[0], pointMemory[1], (pointMemory[1][0], height), (pointMemory[2][0], height), pointMemory[2]])]
				cv2.fillPoly(stencil2, contours, (255, 255, 255))
				stencil = cv2.bitwise_and(stencil1, stencil2)
				mask = cv2.bitwise_and(mask, stencil)
				#cv2.imshow("image", drawings)
				cv2.circle(drawings, pointMemory[0], int(radius), (0, 255, 0), 1)
				
def frameoverlay(original, drawings):

	return cv2.add(original, drawings)

def framemask(original, mask):

	return cv2.bitwise_and(original, mask)

def framecrop(original, mask):

	fm = framemask(original, mask)
	lower = pointMemory[0][1]+int(radius)
	roi = fm[pointMemory[0][1]:lower, pointMemory[1][0]:pointMemory[2][0]]
	return roi


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="Path to the echo video folder")
args = vars(ap.parse_args())


# load the video and play it
cv2.namedWindow("video")


folder = args["folder"]
types = (os.getcwd() + '/' + folder + '/**/*.avi', os.getcwd() + '/' + folder + '/**/*.mp4')
i = 0

for file in types:
	for path in glob.iglob(file, recursive=True):
		patient_id = path.split("/")[-2]
		echo_view = path.split("/")[-1].split(".")[0]
		new_filename = patient_id + '-' + folder + '-' + os.path.basename(path).split('.')[0]

		i += 1
		print(path)
		print(new_filename)

		cap = cv2.VideoCapture(path)
		quit = playvideo(cap, new_filename)

		if quit == True:
			break
			
print("Total files: ", i)

cv2.destroyAllWindows()