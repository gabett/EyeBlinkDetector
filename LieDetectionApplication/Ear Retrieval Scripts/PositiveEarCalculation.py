from scipy.spatial import distance as dist
import dlib
import sys
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from skimage import io
import cv2
import pickle

# Global Variables
positiveEarIntervalsFilePath = 'D:\\COB\\Eyeblink8\\11\EarIntervalsNegative.txt'
frameFilePath = 'D:\Prova\\11\out'


def eye_aspect_ratio(eye):

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	EAR = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return EAR

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


with open(positiveEarIntervalsFilePath) as f:
    lines = f.readlines()
    for line in lines:
        left, right = line.rstrip('\n').split(" ")
        # print('Left:', left, ' Right:', right, '\n')

        positiveEarInterval = []

        frameCounter = 0
        while frameCounter <= 12: 
            
            image_path = frameFilePath + str(int(left) + frameCounter) + '.png'

            frame = io.imread(image_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
            
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                EAR = (leftEAR + rightEAR) / 2.0
                positiveEarInterval.append(EAR)

            frameCounter += 1
        
        with  open('negative_ear11.txt', 'a') as outFile:
            for item in positiveEarInterval:
                outFile.write("%s " % item)
            outFile.write("\n")

        

        

            