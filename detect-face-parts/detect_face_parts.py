from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import face_recognition

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--detection-method", type=str, default='hog',
                help="face detection model to use: either 'hog' or 'cnn' ")

args = vars(ap.parse_args())

# initialize dlibs face detector (HOG-based) and then create the facial landmark predictor
# HOG: histogram of oriented gradients
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# before we can actually detect facial landmarks we first need to detect the
# face in our input image
# load the input image, resize it and convert to grayscale
image = cv2.imread(args['image'])
if image is None:
	raise Exception(f"Could not find image at path: {args['image']}")

print(f"Image shape: {image.shape}")
image = imutils.resize(image, width=500)
print(f"Resized Image shape: {image.shape}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
# The benefit of increasing the resolution of the input image prior to face detection is that it
# may allow us to detect more faces in the image — the downside is that the larger the input image,
# the more computaitonally expensive the detection process is.

rects = hog_face_detector(gray, 1)
print("dlib Number of faces detected: {}".format(len(rects)))
if len(rects) > 0:
	print(rects[0])
	print(rects[0].top())
	print(rects[0].right())
	print(rects[0].bottom())
	print(rects[0].left())


#rects = dlib.rectangles()

"""
I have seen where dlib did not detect some of my pictures.. not sure why so I added
face_recognition to see if I could improve the face recognition portion.
"""

# box(top,right,bottom,left), cv2.rectangle wants (upper left, and bottom right)
#							  cv2.rectangle(image, (b[3],b[0]), (b[1],b[2]),  (255,0,0),2)
boxes = face_recognition.face_locations(gray, model=args['detection_method'])
print("face_recognition Number of faces detected: {}".format(len(boxes)))
if len(boxes) > 0:
	print(boxes[0])
	# dlib.rectangle(left=329, top=78, right=437, bottom=186)
	for b in boxes:
		rects.append(dlib.rectangle(left=b[3], top=b[0], right=b[1], bottom=b[2]))

# Given the (x, y)-coordinates of the faces in the image, we can now apply
# facial landmark detection to each of the face regions
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x,y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# loop over the face parts individually
	for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then display
		# the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

		# loop over teh subset of facial landmarks, drawing the specific face parts
		for (x,y) in shape[i:j]:
			cv2.circle(clone, (x,y), 1, (0,0,255), -1)

			#extract the ROI of the face region as a separate image
			(x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y+h, x:x+w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

			# show the particular face part
			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)
			cv2.waitKey(0)
		output = face_utils.visualize_facial_landmarks(image, shape)
		cv2.imshow("Image", output)
		cv2.waitKey(0)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e. (x,y,w,h)] then draw the face bounding box
	(x,y,w,h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

	# show the face number
	cv2.putText(image, f"Face #{i}", (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

	# loop over the (x,y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x,y) in shape:
		cv2.circle(image, (x,y), 1, (0,0,255), -1)

if len(boxes) > 0:
	pts = boxes[0]
	cv2.rectangle(image, (pts[3],pts[0]), (pts[1], pts[2]), (255,0,0),2)

cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
