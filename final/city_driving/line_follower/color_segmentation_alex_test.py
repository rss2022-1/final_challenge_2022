from turtle import down
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
	# if img == None:
	# 	return None
	w = img.shape[1]
	cropped_im = img
	cv2.rectangle(cropped_im, (0,0), (w, y_cutoff), (0, 0, 0), -1)
	w_offset = 200
	h_offset = 40 + y_cutoff
	r_corner = (0, y_cutoff)
	r_down_corner = (0, h_offset)
	r_w_corner = (w_offset, y_cutoff)
	r_triangle_cnt = np.array([r_corner, r_down_corner, r_w_corner])
	cv2.drawContours(cropped_im, [r_triangle_cnt], 0, (0,0,0), -1)
	l_corner = (w, y_cutoff)
	l_down_corner = (w, h_offset)
	l_w_corner = (w - w_offset, y_cutoff)
	l_triangle_cnt = np.array([l_corner, l_down_corner, l_w_corner])
	cv2.drawContours(cropped_im, [l_triangle_cnt], 0, (0,0,0), -1)

	# Change color space to HSV
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Erode TODO: RE-TUNE THESE VALUES
	img = cv2.erode(img, np.ones((8, 8), 'uint8'), iterations=1)
	img = cv2.dilate(img, np.ones((16,16), 'uint8'), iterations=1)

	# Filter HSV values to get one with the cone color, creating mask while doing so
	orange_min = np.array([10, 80, 100], np.uint8)
	orange_max = np.array([30, 255, 255], np.uint8)
	mask = cv2.inRange(hsv_img, orange_min, orange_max)
	# sensitivity = 80
	# lower_white = np.array([0,0,255-sensitivity])
	# upper_white = np.array([255,sensitivity,255])
	# mask = cv2.inRange(hsv_img, lower_white, upper_white)
	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return mask

def get_contours(src):
	dst = cv2.Canny(src, 50, 200, None, 3)
	# Copy edges to the images that will display the results in BGR
	cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

	linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 10, None, 70, 500)

	prev_lines = [] # (theta, x1, y1, x2, y2, v)

	if linesP is not None:
		for i in range(0, len(linesP)):
			x1, y1, x2, y2 = linesP[i][0]
			v = np.array([x2-x1, y2-y1])
			th = np.arctan2(v[1], v[0])
			pixel_epsilon = 80
			scaled_v = 200 * (v / np.linalg.norm(v))
			# Delete lines that are too horizontal
			# if np.abs(th) < .15:

			# 	continue
			# Delete lines that are similar to previous lines
			if any([(np.linalg.norm(scaled_v - prev_line[5]) < pixel_epsilon) for prev_line in prev_lines]):
				continue
			# new_x2, new_y2 = get_intersection(x2, y2, x1, y1, w, h)
			prev_lines.append([th, x1, y1, x2, y2, scaled_v])
			cv2.line(cdstP, (x1, y1), (x2, y2), (0,0,255), 3, cv2.LINE_AA)
	return cdstP, prev_lines

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return 0, 0

def find_lookahead_point(lane_segments):
    if len(lane_segments) == 0:
        print("No lane segments found")
        return (0,0,0)
    px_lookahead = 250
    lookahead_line = line((0, px_lookahead), (1000, px_lookahead))
    intersections = []
    for i in range(len(lane_segments)):
        l = line(lane_segments[i][1:3], lane_segments[i][3:5])
        x, y = intersection(lookahead_line, l)
        intersections.append((x, y))
    result = np.mean(intersections, axis=0)
    return (int(result[0]), int(result[1]))
        # intersections.append((x, y))
    # if len(intersections) == 1:
    #     print("One intersection found")
    #     return (intersections[0][0], px_lookahead, 0)
    # else:
    #     print(str(len(intersections)) + " intersections found")
    #     return (int((intersections[0][0] + intersections[1][0])/2.), px_lookahead, 0)

def test_segmentation():
	base_path = os.path.dirname(os.getcwd()) + "/line_follower/test_lines/stop"
	end = 7
	# base_path = os.path.abspath(os.getcwd()) + "/test_curve_low_speed/"
	# end = 17
	# base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve/"
	# end = 10
	# base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve_2/"
	# end = 24

	for i in range(1, end):
		print(base_path + str(i) + ".png")
		img = cv2.imread(base_path + str(i) + ".png")
		mask = cd_color_segmentation(img,195)
		image_print(img)
		image_print(mask)
		cdstP, lines = get_contours(mask)
		image_print(cdstP)
		lookahead_point = find_lookahead_point(lines)
		image = cv2.circle(cdstP, (lookahead_point[0], lookahead_point[1]), radius=4, color=(255, 0, 0), thickness=-1)
		print(i)
		image_print(image)
		# cv2.imwrite("masks/mask" + str(i) + ".jpg", mask)



