from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import platform

"""
$ python my_detect_blinks.py --shape-predictor ./blink-detection/shape_predictor_68_face_landmarks.dat --video ./blink-detection/blink_detection_demo.mp4
"""


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x,y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x,y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect rati0
    ear = (A + B) / (2.0 * C)

    return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# ----------------------------------------------------------
# These next 2 variables will need to tuned
# EYE_AR_THRESHOLD: eye aspect ratio threshold to determine if its a blink
#   if the eye aspect ratio falls below a certain threshold and then rises above the
#   threshold, then we register a 'blink'
# EYE_AR_CONSEC_FRAMES: the number of consecutive frames the eye must be below the threshold
#   how many consecutive frames the EAR is below the threshold for a blink to be registered
# -----------------------------------------------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# intialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
# dlib library uses a pre-trained face detector which is based on a modification to the HOG+SVM
# method for object detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# start the video Stream thread
print("[INFO] starting video stream thread...")
video_file = args['video']
file_stream = False
if video_file:
    vs = FileVideoStream(video_file).start()
    file_stream = True
elif platform.system() == 'Linux':
    vs = VideoStream(usePiCamera=True).start()
else:
    vs = VideoStream(src=0).start()


def process_frame(frame):
    global COUNTER, TOTAL
    if frame is None:
        return frame

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over each of the faces in the frame and then apply facial landmark detection to each
    for rect in rects:

        # determine the facial landmarks for the face region, then convert the facial landmarks
        # (x,y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates
        # to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # visualize the facial landmarks
        # compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame
        # counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # otherwise, the eye aspect ratio is not below the blink threshold
            # if the eyes were closed for a sufficient number of frames then increment
            # the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# loop over frames from video stream
while True:

    # if this is a file video stream, then we need to check if there are any more
    # frames left in the buffer to process
    if file_stream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize it, and convert it to grayscale
    frame = vs.read()
    frame = process_frame(frame)

    # show the frame
    if frame is not None:
        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
