
 
import os
print(os.getcwd())
import rospy
from std_msgs.msg import String

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
import math
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import cv2
import numpy as np
#from goprocam import GoProCamera
#from goprocam import constants
vidcap = cv2.VideoCapture('test5.mp4')
#vidcap = cv2.VideoCapture("udp://10.5.5.9:8554")
success,gopro = vidcap.read()
success = True
i=0

lx1 = 0
lx2 = 0
ly1 = 0
ly2 = 0
rx1 = 0
rx2 = 0
ry1 = 0
ry2 = 0
present = 0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform to the grayscaled image"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Initializing empty arrays
    right_xpoints=[]
    right_ypoints=[]
    l_xpoints=[]
    l_ypoints=[]

    sloper = []
    slopel = []
    slopes = []

    br = []
    bl = []
    bs = []
    global present, lx1 , lx2 , ly1 , ly2 , rx1 , rx2 , ry1 , ry2
    if lines is None:
        #left
        print('lol')
        cv2.line(img, (lx1, ly1), (lx2, ly2), [0, 0, 255], 6)
        #right
        cv2.line(img, (rx1, ry1), (rx2, ry2), [0, 0, 255], 6)
        
        #avgx1, avgy1, avgx2, avgy2 = avgLeft
        #cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255,255,255], 8) #draw left line
        #avgx3, avgy3, avgx4, avgy4 = avgRight
        #cv2.line(img, (int(avgx3), int(avgy3)), (int(avgx4), int(avgy4)), [255,255,255], 8) #draw right line
        #cv2.line(img, (int((int(avgx3)+int(avgx1))/2), int((int(avgy3)+int(avgy1))/2)), (int((int(avgx4)+int(avgx2))/2), int((int(avgy4)+int(avgy2))/2)), [255,255,255], 8) #draw right line
        return
    
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            #Finding the slope and intercept value
            slope = ((y2-y1)/(x2-x1))
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            
            #Filter the lines based on the slope value as right and left lanes
            if (slope > 0.3 and ((x1 and x2) > 320)):
              #right lane
                presentl = 1
                sloper.append(slope)
                br.append(parameters[1])
                #Adding right hough lines
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.3 and ((x1 and x2) < 320)): 
                #left lane
                presentr = 1
                slopel.append(slope)
                bl.append(parameters[1]) 
                #Adding left hough lines
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
#            elif (-0.2 < slope < 0.2 and ((y1 and y2 )< 350 )):
#                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    #Calculating mean slope and intercept values
    
    #right lane
    meanSloper = np.mean(sloper) 
    meanBr = np.mean(br)
    
    if (present == 0):
         cv2.line(img, (rx1, ry1), (rx2, ry2), [0, 0, 255], 6)
         cv2.line(img, (lx1, ly1), (lx2, ly2), [0, 0, 255], 6)
    #finding two points to fit the right lane line
    x1 = 640
    if(np.isnan(meanSloper) or np.isnan(meanBr)):
      pass
    else:
      
        y1 = meanSloper * x1 + meanBr
        y2 = 260
        x2 = ( y2 - meanBr ) / meanSloper
        #Plotting the right lane
        if(math.isfinite(y1) and math.isfinite(x2)):
            cv2.line(img, (x1, int(y1)), (int(x2), y2), [0, 0, 255], 6)
        #print('lol')
        
        #right
       
        lx1 = x1 
        lx2 = int(x2)
        ly1 = int(y1)
        ly2 = y2
    #left lane    
    meanSlopel = np.mean(slopel) 
    meanBl = np.mean(bl)
    
    #finding two points to fit the left lane line
    x1 = 0
    if(np.isnan(meanSlopel) or np.isnan(meanBl) or math.isinf(meanSlopel) or math.isinf(meanBl)):
      pass
    else:    
        y1 = meanSlopel * x1 + meanBl
        y2 = 260
        x2 = ( y2 - meanBl ) / meanSlopel
        #Plotting the left lane
        if(math.isfinite(y1) and math.isfinite(x2)):
            cv2.line(img, (x1, int(y1)), (int(x2), y2), [0, 0, 255], 6)   
       
        rx1 = x1 
        rx2 = int(x2)
        ry1 = int(y1)
        ry2 = y2
        
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Takes the outout of canny as the input image and
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    Takes the output image with hough lines as input

    The result image is computed as follows:
    
    initial_img * a + img * ß + ?
    """
    return cv2.addWeighted(initial_img, a, img, b, c)

# Import everything needed for creating the video clips


def process_image(image):
    # Resizing the input image to a more resonable size for processing
    

    
    
    
    
    # Blur to avoid edges from noise
    blurredImage = gaussian_blur(image, 7)
    
    # Detect edges using canny
    edgesImage = canny(blurredImage, 100, 140)
    
    # Mark out the vertices for region of interest
    vertices = np.array( [[
                [0, 480],
                [0, 320],
                [200, 280],
                [520, 260],
                [640, 300],
                [640, 480]
            ]], dtype=np.int32 )
 
    # Mask the canny output with region of interest
    regionInterestImage = region_of_interest(edgesImage, vertices)
    
    # Drawing the hough lines in the Masked Canny image
    lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi/180, 35, 15, 100)
    
    # Test detected edges by uncommenting this
    # return cv2.cvtColor(regionInterestImage, cv2.COLOR_GRAY2RGB)

    # Draw output on top of original
    return weighted_img(lineMarkedImage, image)

rospy.init_node('ObiWan')
pub = rospy.Publisher('chatter', String, queue_size=10)
pubssd = rospy.Publisher('chatterssd', String, queue_size=10)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
(startX, startY, endX, endY) = (0,0,0,0)
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
#rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown() and success :
	i=i+1 
	
	presentr = 0
	presentl = 0
	if gopro is not None:

		gopro = cv2.resize(gopro, (640, 480), interpolation = cv2.INTER_AREA)
		image = process_image(gopro)
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		print("[INFO] computing object detections...")
		net.setInput(blob)
		detections = net.forward()

	    # loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > 0.4:
				# extract the index of the class label from the `detections`,
				# then compute the (x, y)-coordinates of the bounding box for
				# the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# display the prediction
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				print("[INFO] {}".format(label))
				cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				pubssd.publish(str(startX))
		cv2.imshow('frame',image)
		#cv2.imwrite("frame"+str(i)+".jpg", mah) 
		key = cv2.waitKey(1)
		if(key == ord('q')):
		  break
	success,gopro = vidcap.read()
	a = str(lx1) + " " +str(lx2) + " " + str(ly1) + " " + str(ly2)  + " " + str(rx1) 
	pub.publish(a)
	
	#print(lx1 , lx2 , ly1 , ly2 , rx1 , rx2 , ry1 , ry2)
	#rate.sleep()

vidcap.release()
cv2.destroyAllWindows()
