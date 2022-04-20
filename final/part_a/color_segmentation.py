import cv2
import numpy as np
import pdb
import os

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

def cd_color_segmentation(img, y_cutoff=0):
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
	w = img.shape[1]
	cropped_im = img
	cv2.rectangle(cropped_im, (0,0), (w, y_cutoff), (0, 0, 0), -1)

	# Change color space to HSV
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Erode TODO: RE-TUNE THESE VALUES
	img = cv2.erode(img, np.ones((8, 8), 'uint8'), iterations=1)
	img = cv2.dilate(img, np.ones((16,16), 'uint8'), iterations=1)

	# Filter HSV values to get one with the cone color, creating mask while doing so
	sensitivity = 25
	lower_white = np.array([0,0,255-sensitivity])
	upper_white = np.array([255,sensitivity,255])
	mask = cv2.inRange(hsv_img, lower_white, upper_white)
	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return mask

def test_segmentation():
	base_path = os.path.abspath(os.getcwd()) + "/test_imgs/img_"
	for i in range(4):
		img = cv2.imread(base_path + str(i) + ".jpg")
		mask = cd_color_segmentation(img,250)
		image_print(img)
		image_print(mask)
		if i == 0:
			cv2.imwrite("mask.jpg", mask)

test_segmentation()


