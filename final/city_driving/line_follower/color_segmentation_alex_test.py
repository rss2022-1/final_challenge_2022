from turtle import down
import cv2
import numpy as np
import pdb
import os
from homography_transformer import HomographyTransformer

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
	if img is None:
		return None
	h,w = img.shape[:2]
	start_y = 200
	end_y = 330
	cropped_im = img.copy()
	cv2.rectangle(cropped_im, (0,0), (w, start_y), (255, 255, 255), -1)
	cv2.rectangle(cropped_im, (0, end_y), (w, h), (255, 255, 255), -1)	
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
	hsv_img = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2HSV)

	# Erode TODO: RE-TUNE THESE VALUES
	# hsv_img = cv2.erode(hsv_img, np.ones((8, 8), 'uint8'), iterations=1)
	# hsv_img = cv2.dilate(hsv_img, np.ones((16,16), 'uint8'), iterations=1)

	# Filter HSV values to get one with the cone color, creating mask while doing so
	orange_min = np.array([10, 80, 80], np.uint8)
	orange_max = np.array([32, 255, 255], np.uint8)
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

	linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 10, None, 80, 500)

	prev_lines = [] # (theta, x1, y1, x2, y2, v)

	if linesP is not None:
		for i in range(0, len(linesP)):
			u1, v1, u2, v2 = linesP[i][0]
			v = np.array([u2-u1, u2-v1])
			th = np.arctan2(v[1], v[0])
			pixel_epsilon = 500
			scaled_v = 1000 * (v / np.linalg.norm(v))
			# Delete lines that are too horizontal
			# if np.abs(th) < .15:

			# 	continue
			# Delete lines that are similar to previous lines
			if any([(np.linalg.norm(scaled_v - prev_line[5]) < pixel_epsilon) for prev_line in prev_lines]):
				continue
			# new_u2, new_y2 = get_intersection(x2, y2, x1, v1, w, h)
			prev_lines.append([th, u1, v1, u2, v2, scaled_v])
			cv2.line(cdstP, (u1, v1), (u2, v2), (0,0,255), 3, cv2.LINE_AA)
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

def find_lookahead_point(lane_segments, w, h):
    # if len(lane_segments) == 0:
    #     print("No lane segments found")
    #     return (0,0,0)
    # px_lookahead = 250
    # lookahead_line = line((0, px_lookahead), (1000, px_lookahead))
    # intersections = []
    # for i in range(len(lane_segments)):
    #     l = line(lane_segments[i][1:3], lane_segments[i][3:5])
    #     x, y = intersection(lookahead_line, l)
    #     intersections.append((x, y))
    # result = np.mean(intersections, axis=0)
    # return (int(result[0]), int(result[1]))
	# import pdb; pdb.set_trace()
	center = np.array([w//2, h])
	lookahead = 200
	intersections = []
	ts = []
	for th, x1, y1, x2, y2, _ in lane_segments:
		p1 = np.array([x1, y1])
		p2 = np.array([x2, y2])
		V = p2 - p1
		a = V.dot(V)
		b = 2 * V.dot(p1-center)
		c = p1.dot(p1) + center.dot(center) - 2 * p1.dot(center) - lookahead**2
		disc = b**2 - 4 * a * c
		if disc < 0:
			continue
		else:
			sqrt_disc = np.sqrt(disc)
			t1 = (-b + sqrt_disc) / (2 * a)
			t2 = (-b - sqrt_disc) / (2 * a)
			if 0 <= t1 <= 1 and 0 <= t2 <= 1:
				t = np.mean([t1, t2], axis=0)
			elif 0 <= t1 <= 1:
				t = t1
			elif 0 <= t2 <= 1:
				t = t2
			else:
				t = t1
			result =  p1 + t * V
			print("p1 + t * V = ", p1 + t * V)
			intersections.append(p1 + t * V)
			ts.append(t)
	direct_intersections = []
	for i in range(len(intersections)):
		point, t = intersections[i], ts[i]
		if t <= 1 and t >= 0:
			direct_intersections.append((int(point[0]), int(point[1])))
	if len(direct_intersections) > 0:
		result = np.mean(direct_intersections, axis=0)
	elif len(intersections) == 0:
		result = (0, 0)
	else:
		result = np.mean(intersections, axis=0)
	return (int(result[0]), int(result[1]))


def test_segmentation():
	base_path = os.path.dirname(os.getcwd()) + "/line_follower/test_lines/stop"
	end = 9
	# base_path = os.path.abspath(os.getcwd()) + "/test_curve_low_speed/"
	# end = 17
	# base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve/"
	# end = 10
	# base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve_2/"
	# end = 24

	for i in range(4, end):
		print(base_path + str(i) + ".png")
		img = cv2.imread(base_path + str(i) + ".png")
		image_print(img)
		mask = cd_color_segmentation(img,195)
		image_print(mask)
		cdstP, lines = get_contours(mask)
		image_print(cdstP)
		h, w = cdstP.shape[:2]
		print(cdstP.shape)
		lookahead_point = find_lookahead_point(lines, w, h)
		cv2.circle(cdstP, (w//2, h), radius=200, color=(0, 255, 255), thickness=4)
		cv2.circle(cdstP, (lookahead_point[0], lookahead_point[1]), radius=20, color=(255, 255, 0), thickness=-1)

		image_print(cdstP)
		# cv2.imwrite("masks/mask" + str(i) + ".jpg", mask)

test_segmentation()


