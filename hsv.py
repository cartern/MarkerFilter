#Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV);


#cv2.cvtColor(src, code[, dst[, dstCn]]) => dst
#from __future__ import print_function
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

marker = False


def hsv(image):	
	global theta
	global x
	global y
	img = cv2.cvtColor(image,  cv2.COLOR_BGR2HSV)
	lower = np.array([4, 0, 100])
	upper = np.array([41, 255, 255])
	pic = cv2.inRange(img, lower, upper)
	output = pic.copy()
	#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) <=  EXAMPLE
	
	contours = cv2.findContours(pic,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
	#print contours checking for errors		<= DIDNT WORK
	#cv2.drawContours(img, contours, -1, (0,255,0), 3)	<= EXAMPLE

	#ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)	<= DIDNT WORK


	#for cnt in ctr :			<=THIS ONE WORKED
	#	print cv2.contourArea(cnt)	<=JUST KIDDING, IT DIDNT
 
	
	for contour in contours:
		area = cv2.contourArea(contour)	
		if area < 3000:
			continue
		rect = cv2.minAreaRect(contour)
		theta = rect[2]
		dim  = rect[1]
		length = dim[1]
		width = dim[0]
		xy = rect[0]
		x = xy[0]
		y = xy[1]
		aspectRatio = length/width
		rectangular = (area / (length * width))
		if aspectRatio >= 3  and aspectRatio <= 12 and rectangular > .6:
			print "angle: " + str(theta)		#check
			print "length: " + str(length)		#check	
			print "width: " + str(width)		#check
			print "aspectRatio: " + str(aspectRatio)	#check
			print "rectangular: " + str(rectangular)	#check
			cv2.drawContours(image, [contour], 0, (0,255,0), 3)
			marker = True
		else:
			marker = False
			continue 
		
		
		
	
	return output
#	cv2.imshow("window",image)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()


image_pub = rospy.Publisher("Image_Published", Image, queue_size=10)
marker_present = rospy.Publisher("Marker_Found", Bool, queue_size = 10) 
angle  = rospy.Publisher("Angle", Float64, queue_size = 10)
x_center = rospy.Publisher("X_Center", Float64, queue_size = 10)
y_center = rospy.Publisher("Y_Center", Float64, queue_size = 10)


bridge = CvBridge()

def callback(data):
	try:
		image = bridge.imgmsg_to_cv2(data, "bgr8")
		result = hsv(image)
		data_published = bridge.cv2_to_imgmsg(result)
		image_pub.publish(data_published)
		marker_present.publish(marker)
		angle.publish(theta)
		x_center.publish(x)
		y_center.publish(y)
	except CvBridgeError as e:
		print(e)

rospy.init_node('Floor_Filter_Motherfucker', anonymous=True)
image_sub = rospy.Subscriber("cv_camera/image_raw", Image,callback)

#cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
#image = cv2.imread('img9.png')
rospy.spin()
#cv2_to_compressed_imgmsg(cvim, dst_format='jpg')

