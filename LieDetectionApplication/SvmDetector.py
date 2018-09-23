import Classifier
import imutils
from imutils.video import FileVideoStream
import time
import dlib
import cv2
from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

class SvmDetector:

    def __init__(self):
        self.isRunning = True
        self.frameNumber = 0
        self.linearSvc = Classifier.Classifier()
        self.linearSvc.trainEyeblink8()
        self.blinks = 0
        self.frames = []
        self.Xs = []
        self.Ys = []
        self.OutcomeValues = [0 for i in range(13)]
        self.interBlinkTimes = []
        self.blinkFramesGap = 0
        self.previousBlinkTimeStamp = 0
        self.nextBlinkTimeStamp = 0
        self.blinksPerMinute = 0
        self.interBlinksTime = 0
        self.avgInterBlinksTime = 0
        self.startTime = 0
        self.endTime = 0
        self.std = 0

    def InitPlot(self):
        print('Creating plot')
        QtGui.QApplication.processEvents()
        self.app = QtGui.QApplication([])
        self.mainLayout = pg.GraphicsLayout(border=(100,100,100))
        self.mainLayout.nextRow()

        self.earPlotLayout = self.mainLayout.addLayout(colspan = 3)
        self.earPlotLayout.setContentsMargins(10,10,10,10)
        self.earPlotLayout.addLabel("EAR plot", colspan = 3)
        self.earPlotLayout.nextRow()
        self.earPlotLayout.addLabel('EAR', angle = -90, rowspan = 2)
        self.earPlot = self.earPlotLayout.addPlot()
        self.earPlotLayout.nextRow()
        self.earPlotLayout.addLabel('Frame', col = 1, colspan = 2)

        self.mainLayout.nextRow()

        self.blinkNoBlinkPlotLayout = self.mainLayout.addLayout(colspan = 3)
        self.blinkNoBlinkPlotLayout.setContentsMargins(10,10,10,10)
        self.blinkNoBlinkPlotLayout.addLabel("Blink / No Blink plot", colspan = 3)
        self.blinkNoBlinkPlotLayout.nextRow()
        self.blinkNoBlinkPlotLayout.addLabel('Blink', angle = -90, rowspan = 2)
        self.blinkNoBlinkPlot = self.blinkNoBlinkPlotLayout.addPlot()
        self.blinkNoBlinkPlotLayout.nextRow()
        self.blinkNoBlinkPlotLayout.addLabel('Frame', col = 1, colspan = 2)

        self.view = pg.GraphicsView()
        self.view.setCentralItem(self.mainLayout)
        self.view.show()
        self.view.setWindowTitle('Plots')
        self.view.resize(800,600)
        print('Done!')

    def eye_aspect_ratio(self, eye):
        '''
        Eye Aspect Ratio (EAR) calculation function
        '''
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
 
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
 
        # return the eye aspect ratio
        return ear

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
            file.write("%s "% item)

        print('Data correctly saved to file.')
        file.close()

    def Start(self, filename):
        '''
        Parameters:
        `filename` path of the video file
        '''

        self.shape_predictor = ".\\shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        
        self.vs = FileVideoStream(filename).start()
        fileStream = True
        time.sleep(1)

        print("Arranging video frames ...")
        while self.vs.more():
            frame = self.vs.read()
            # Transpose and flip statements performed everytime the orientation
            # of
            # the video passed is not
            # initially correct.
            #frame = cv2.transpose(frame)
            frame = cv2.flip(frame, flipCode=1) # HACK: this tweak depends on the original video orientation.

            frame = imutils.resize(frame, width=350)
            self.frames.append(frame)

        print('Done!')

        self.InitPlot()
        self.ReadFrames()

    def ReadFrames(self):

        self.left_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] # tuple (lStart, lEnd)
        self.right_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # tuple (rStart, rEnd)
        self.mouth_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["mouth"] # tuple (mStart, mEnd)
		# extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        
        self.startTime = time.time()

        for frame in self.frames:
            self.frames[self.frameNumber] = None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            # loop over the face detections
            for rect in rects:

		        # determine the facial landmarks for the face region, then
		        # convert the facial landmark (x, y)-coordinates to a NumPy
		        # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
 
                leftEye = shape[self.left_landmarks[0]:self.left_landmarks[1]]
                rightEye = shape[self.right_landmarks[0]:self.right_landmarks[1]]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
 
		        # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                self.Xs.append(self.frameNumber)
                self.Ys.append(ear)

                if(self.frameNumber >= 13):
                    if self.blinkFramesGap == 0:
                        outcome = self.linearSvc.classify(self.Ys[len(self.Ys)-13:])
                        
                        if outcome == 1:
                            self.OutcomeValues.append(1)
                            self.blinks += 1
                            self.blinkFramesGap = 13

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
                    else:
                        self.blinkFramesGap -=1
                        self.OutcomeValues.append(0)

                if len(self.Xs) % 30 == 0:
                    self.blinkNoBlinkPlot.plot(self.Xs, self.OutcomeValues, pen='w', clear = True)
                    self.earPlot.plot(self.Xs, self.Ys, pen='r', clear = True)

		        # compute the convex hull for the left and right eye, then
		        # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		        # draw the total number of blinks on the frame along with
		        # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(self.blinks), (10, 30),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (230, 30),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Blinks per minute: {:.2f}".format(self.blinksPerMinute), (10, 500),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Avg inter blink time: {:.2f}".format(self.avgInterBlinksTime), (10, 600),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # show the frame
            cv2.imshow("Frame", frame)

            # read key
            key = cv2.waitKey(1) & 0xFF
 
	        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
 
            self.frameNumber +=1

## Post streaming operations.

        self.endTime = time.time()
        self.isRunning = False
        # Displays the standard deviation.
        self.std = np.sqrt(np.var(self.interBlinkTimes))
        print("Standard deviation : {:.2f}".format(self.std))
        print("Number of blinks: {:.2f}".format(self.blinks))
        print("Video length: {:.2f}".format(self.endTime - self.startTime))

        # Closes all windows.
        cv2.destroyAllWindows()
        self.vs.stop()
        self.view.close()
        self.SaveDataToFile()

        return 0
