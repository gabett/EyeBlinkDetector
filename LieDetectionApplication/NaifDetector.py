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



class NaifDetector:

    def  __init__(self):
        self.frames = []
        self.Xs = []
        self.Ys = []
        self.interBlinkTimes = []
        self.std = 0
        self.blinks = 0
        self.startTime = 0
        self.endTime = 0
        self.frameNumber = 0
        self.OutcomeValues = [0 for i in range(13)]
        self.blinkFramesGap = 0
        self.previousBlinkTimeStamp = 0
        self.nextBlinkTimeStamp = 0
        self.blinksPerMinute = 0
        self.interBlinksTime = 0
        self.avgInterBlinksTime = 0
        self.endTime = 0

    def SaveDataToFile(self):
        print('Saving data to file ...')

        file = open('.\\Data\\Results.txt', 'w')
        file.write("Video Length:\n%s\n" % (self.endTime - self.startTime))
        file.write("Frames number:\n%s\n" % len(self.Xs))
        file.write('Standard Deviation:\n%s\n' % self.std)
        file.write('Number of blinks:\n%s\n' % self.blinks)

        file.write('EARs:\n')
        for item in self.Ys:
            file.write("%s " % item)
        
        file.write('\nSeconds between blinks:\n')
        for item in self.interBlinkTimes:
            file.write("%s " % item)

        file.write('\nBlink-No Blink:\n')

        for item in self.OutcomeValues:
            file.write("%s " % item)

        print('Data correctly saved to file.')
        file.close()

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def Start(self, filepath):
        EYE_AR_THRESH = 0.4
        EYE_AR_CONSEC_FRAMES = 3

        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        TOTAL = 0

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        shape_predictor = ".\\shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        print("[INFO] starting video stream thread...")
        self.vs = FileVideoStream(filepath).start()
        fileStream = True
        time.sleep(1.0)

        print("Arranging video frames ...")
        while self.vs.more():
            frame = self.vs.read()
            # Transpose and flip statements performed everytime the orientation
            # of
            # the video passed is not
            # initially correct.
            #frame = cv2.transpose(frame)
            frame = cv2.flip(frame, flipCode=1) # HACK: this tweak depends on the original video orientation.

            frame = imutils.resize(frame, width=500)
            self.frames.append(frame)

        print('Done!')

        self.startTime = time.time()

        # loop over frames from the video stream
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frameNumber += 1

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                self.Xs.append(self.frameNumber)
                self.Ys.append(ear)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        self.blinks += 1
                        self.OutcomeValues.append(1)

                        if self.previousBlinkTimeStamp == 0:
                            self.previousBlinkTimeStamp = time.time()
                        else:
                            self.nextBlinkTimeStamp = time.time()
                            self.interBlinksTime = self.nextBlinkTimeStamp - self.previousBlinkTimeStamp
                            self.interBlinkTimes.append(self.interBlinksTime)
                            self.blinksPerMinute = 60 / self.interBlinksTime

                            if self.avgInterBlinksTime == 0:
                                self.avgInterBlinksTime = self.interBlinksTime

                            self.avgInterBlinksTime = (self.avgInterBlinksTime + self.interBlinksTime) / 2
                            self.previousBlinkTimeStamp = self.nextBlinkTimeStamp
                         
                            self.nextBlinkTimeStamp = 0

                    else:
                        self.OutcomeValues.append(0)


                    # reset the eye frame counter
                    COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(self.blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        self.endTime = time.time()
        isRunning = False
        # Displays the standard deviation.
        self.std = np.sqrt(np.var(self.interBlinkTimes))
        print("Standard deviation : {:.2f}".format(self.std))
        print("Number of blinks: {:.2f}".format(self.blinks))
        print("Video length: {:.2f}".format(self.endTime - self.startTime))
        # do a bit of cleanup
        self.SaveDataToFile()
        cv2.destroyAllWindows()
        self.vs.stop()