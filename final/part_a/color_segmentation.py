import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################
def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	bounding_box = ((0,0),(0,0))

	# Change color space to HSV
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Erode TODO: RE-TUNE THESE VALUES
	kernel = np.ones((4, 4), 'uint8')
	img = cv2.erode(img, kernel, iterations=1)
	img = cv2.dilate(img, np.ones((8,8), 'uint8'), iterations=1)

	# Filter HSV values to get one with the cone color, creating mask while doing so
	ORANGE_MIN = np.array([5, 170, 170],np.uint8) # [Hue, Saturation, Value] #5/17 # 5
	ORANGE_MAX = np.array([100, 255, 255],np.uint8) # 17
	mask = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
	# image_print(mask)

	# Put mask through
	contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	max_h, max_w, best_x, best_y = 0, 0, 0, 0
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if w * h > max_w * max_h:
			max_h = h
			max_w = w
			best_x = x
			best_y = y

	cv2.rectangle(mask,(best_x,best_y),(best_x+max_w,best_y+max_h),(255,0,0),1)

	bounding_box = ((best_x,best_y),(best_x+max_w,best_y+max_h))

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box, mask
