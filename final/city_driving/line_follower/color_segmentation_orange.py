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
    if img is None:
    	return None
    h,w = img.shape[:2]
    start_y = 200
    end_y = 330
    top_mid_y = (end_y - start_y)//3 + start_y
    bottom_mid_y = 2*(end_y - start_y)//3 + start_y
    cropped_im = img.copy()
    # # Top rectangle
    # cv2.rectangle(cropped_im, (0,0), (w, start_y), (255, 255, 255), -1)
    # # Bottom rectangle
    # cv2.rectangle(cropped_im, (0, end_y), (w, h), (255, 255, 255), -1)
    center_image_x = w//2
    thresh = 100
    # Top rectangle
    cv2.rectangle(cropped_im, (0,0), (w, start_y), (255, 255, 255), -1)
    # Left rectangle
    cv2.rectangle(cropped_im, (0,0), (center_image_x - thresh, top_mid_y), (255, 255, 255), -1)
    # Right rectangle
    cv2.rectangle(cropped_im, (center_image_x + thresh, 0), (w, top_mid_y), (255, 255, 255), -1)
    # Long bottom rectangle
    cv2.rectangle(cropped_im, (0, end_y), (w, h), (255, 255, 255), -1)
    # Small bottom rectangle
    cv2.rectangle(cropped_im, (center_image_x - thresh, bottom_mid_y), (center_image_x + thresh, h), (255, 255, 255), -1)
    # image_print(cropped_im)

    # Change color space to HSV
    hsv_img = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2HSV)

    # Erode TODO: RE-TUNE THESE VALUES
    #cropped_im = cv2.erode(cropped_im, np.ones((8, 8), 'uint8'), iterations=1)
    cropped_im = cv2.dilate(cropped_im, np.ones((50,50), 'uint8'), iterations=1)

    # Filter HSV values to get one with the orange line color, creating mask while doing so
    sensitivity = 80
    # [Hue, Saturation, Value]
    # Bright orange
    # orange_min = np.array([5, 170, 170],np.uint8)
    # orange_max = np.array([100, 255, 255],np.uint8)
    # Light orange
    orange_min = np.array([1, 80, 100],np.uint8)
    orange_max = np.array([255, 255, 255],np.uint8)
    mask = cv2.inRange(hsv_img, orange_min, orange_max)
    
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_h, max_w, best_x, best_y = 0, 0, 0, 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w * h > max_w * max_h:
            max_h = h
            max_w = w
            best_x = x
            best_y = y

    cv2.rectangle(mask,(best_x,best_y),(best_x+max_w,best_y+max_h),(255,255,0),3)

    bounding_box = ((best_x,best_y),(best_x+max_w,best_y+max_h))
    # image_print(img)
    # image_print(mask)

    # Return bounding box
    return bounding_box, mask

def test_segmentation():
    base_path = os.path.abspath(os.getcwd()) + "/"
    # end = 8
    # base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve/"
    # end = 10
    # base_path = os.path.abspath(os.getcwd()) + "/test_straight_curve_2/"
    # end = 24

    for i in range(1, 2):
        img = cv2.imread(base_path + 'city_clean' + str(i) + ".png")
        bounding_box, mask = cd_color_segmentation(img)
        # image_print(img)
        # h,w = img.shape[:2]
        # start_y = 250
        # end_y = 330
        # print(start_y, end_y)
        # cv2.rectangle(mask, (0,0), (w, start_y), (255, 255, 255), -1)
        # cv2.rectangle(mask, (0, end_y), (w, h), (255, 255, 255), -1)
        cv2.rectangle(mask, bounding_box[0], bounding_box[1], (255,0,0), 1)
        tlx, tly = bounding_box[0] # top left
        brx, bry = bounding_box[1] # back right
        center_x, center_y = (brx - tlx)/2.0 + tlx, bry
        cv2.circle(mask, (int(center_x), int(center_y)), radius=4, color=(255, 0, 0), thickness=-1)
        image_print(mask)
        # cv2.imwrite(base_path + "/line_masks/mask" + str(i) + ".jpg", mask)

# test_segmentation()


