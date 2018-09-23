from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer
from imutils import face_utils
from math import floor
import subprocess
import threading
import librosa
import numpy as np
import imutils
import time
import dlib
import cv2
import os

SHAPE_PREDICTOR = ".\\shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.03
MOUTH_AR_CONSEC_FRAMES = 5
MAX_FRAMES_TALKING = 3 * MOUTH_AR_CONSEC_FRAMES

ENERGY_TRESHOLD = 50
MIN_LENGTH = 20
MAX_LENGTH = 1000
MAX_CONTINUOUS_SILENCE = 40

class LieDetection:
    '''
    '''

    def __init__(self, video_filename_path, shape_predictor=SHAPE_PREDICTOR):
        self.video_filename_path = video_filename_path
        self.video_file_name = os.path.basename(self.video_filename_path)
        self.raw_file_name = os.path.basename(self.video_file_name).split('.')[0]
        self.file_dir = os.path.dirname(self.video_filename_path)
        self.shape_predictor = shape_predictor 
        self.counter = 0 # Eye frame counter activated whenever Ear is below its own threshold.
        self.counter_mouth = 0 # Mouth frame counter which works similarily as the above counter. 
        self.total = 0 # blinks counter
        self.frameNumber = 0 # frame counter that keeps track of the current number of the frame analyzed
        self.total_baseline_blinks = 0 # total number of baseline blinks
        self.baseline_blinks = 0 # temporary baseline blinks counter
        self.blinks_while_talking = 0 # temporary blinks counter whenever the subject is talking
        self.total_blinks_while_talking = 0 # total blinks counter whenever the subject is talking
        self.final_time = 0 # measures the time when the video has ended.
        self.is_program_ended = False 
        self.has_started_talking = True
        self.is_talking = False
        self.maxBaselineLengthPeriod = 10 # We assume the baseline period is 10 seconds long.
        self.output_data = []


    def initialize_face_detector(self):
        '''
        Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor.
        Grab the indexes of the facial landmarks for the left and right eye, respectively.
        '''
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.left_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] # tuple (lStart, lEnd)
        self.right_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # tuple (rStart, rEnd)
        self.mouth_landmarks = face_utils.FACIAL_LANDMARKS_IDXS["mouth"] # tuple (mStart, mEnd)


    def tokenize_audio(self, record=True):
        '''
        Default analysis window is 10 ms (float(asource.get_block_size()) / asource.get_sampling_rate())
        min_length=20 : minimum length of a valid audio activity is 20 * 10 == 200 ms
        max_length=4000 :  maximum length of a valid audio activity is 400 * 10 == 4000 ms == 4 seconds
        max_continuous_silence=30 : maximum length of a tolerated silence within a valid audio activity is 30 * 30 == 300 ms

        Parameters:
        `record=True` argument means we can rewind the source
        '''
        self.split_audio_from_video()
        asource = ADSFactory.ads(filename=self.audio_filename_path, record=record)
        validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=ENERGY_TRESHOLD)
        tokenizer = StreamTokenizer(validator=validator, min_length=MIN_LENGTH, max_length=MAX_LENGTH, max_continuous_silence=MAX_CONTINUOUS_SILENCE)
        asource.open()
        self.tokens = tokenizer.tokenize(asource)


    def eye_blink_detector(self, filename, show_frames=False):
        '''
        Parameters:
        `filename` path of the video file
        `show_frames=True` argument reproduce video frames
        '''
        vs = FileVideoStream(filename).start()
        fileStream = True
        time.sleep(1.0)

        while not self.is_program_ended:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = vs.read()
            frame = imutils.resize(frame, width=450) # resizing the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
            frameNumber += 1

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            # loop over the face detections
            for rect in rects:
		        # determine the facial landmarks for the face region, then
		        # convert the facial landmark (x, y)-coordinates to a NumPy
		        # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
 
		        # extract the left and right eye coordinates, then use the
		        # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[self.left_landmarks[0]:self.left_landmarks[1]]
                rightEye = shape[self.right_landmarks[0]:self.right_landmarks[1]]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
 
		        # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

		        # compute the convex hull for the left and right eye, then
		        # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		        # check to see if the eye aspect ratio is below the blink
		        # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    self.counter += 1
 
		        # otherwise, the eye aspect ratio is not below the blink
		        # threshold
                else:
			        # if the eyes were closed for a sufficient number of
			        # then increment the total number of blinks
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        self.total += 1
 
			        # reset the eye frame counter
                    self.counter = 0

		        # draw the total number of blinks on the frame along with
		        # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(self.total), (10, 30),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
            # show the frame
            if (show_frames == True):
                cv2.imshow("Frame", frame)

            # read key
            key = cv2.waitKey(1) & 0xFF
 
	        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
 
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def eye_blink_and_mouth_detector(self, filename, show_frames=False): # ASK: main?
        '''
        `show_frames=True` argument reproduce video frames
        '''
        global initialTime
        self.has_started_talking = False
        # Starting the video.
        vs = FileVideoStream(filename, queueSize=1000).start()
        fileStream = True
        time.sleep(1.0)

        # Starting baseline blinks thread.
        t1 = threading.Thread(target=self.baseline_blinks_per_minute, args=[])
        t1.start()

        initialTime = 0
        while not self.is_program_ended:	
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)450
            frame = vs.read()

            # Transpose and flip statements performed whenever the horientation of
            # the video passed is not
            # initially correct.
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, flipCode=-1) # HACK: this tweak depends on the original video orientation.

            frame = imutils.resize(frame, width=450)

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

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[self.left_landmarks[0]:self.left_landmarks[1]]
                rightEye = shape[self.right_landmarks[0]:self.right_landmarks[1]]
                # Extract mouth coordinates used to compute mouth aspect ratio.
                mouth = shape[self.mouth_landmarks[0]:self.mouth_landmarks[1]]

                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                MAR = self.mouth_aspect_ratio(mouth)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                # compute the convex hull for mouth, then visualize it.
                mouthHull = cv2.convexHull(mouth)

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    self.counter += 1

		        # otherwise, the eye aspect ratio is not below the blink
		        # threshold
		        # if the eyes were closed for a sufficient number of
		        # then increment the total number of blinks
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    if self.is_talking:
                        self.blinks_while_talking += 1
                        print("Talking blinks: ", self.blinks_while_talking)

                    self.baseline_blinks += 1
                    print("Baseline blinks: ", self.baseline_blinks)
                    self.counter = 0

		        # The mouth talking counter increases with a 2x speed factor before
		        # its own threshold, then increase with 1x speed factory.
		        # This has been chosen in order to obtain a good balance between
		        # short and long sentences.
                if MAR >= MOUTH_AR_THRESH and self.counter_mouth <= MAX_FRAMES_TALKING:
                    self.counter_mouth += 2
                if MAR >= MOUTH_AR_THRESH and self.counter_mouth <= MAX_FRAMES_TALKING:
                    self.counter_mouth += 1

		        # The mouth talking counter decreseas with 1x speed factor in order
		        # to manage scenarios where short breaks
		        # during sentences are performed, which have not to be recognzed as
		        # sentence periods.
                else:
                    if self.counter_mouth > 0:
                        self.counter_mouth -=1

		        # Whenever the threshold is reached, the talking flag is set to true.
                if self.counter_mouth >= MOUTH_AR_CONSEC_FRAMES:
                    self.is_talking = True

			        # If the subject is not already talking, a initial time of the
			        # sentence is registered and printed out.
                    if self.has_started_talking == False:
                        initialTime = time.time()
                        print("Initial time: ", initialTime)

                    self.has_started_talking = True

                else:
			        # If the mouth talking counter is above the threshold and at the
			        # same time the subject is greater than zero
			        # (which means that the subject was previously talking), post
			        # speaking math is performed.
                    if self.counter_mouth < MOUTH_AR_CONSEC_FRAMES and self.counter_mouth > 0 and self.is_talking:

                        # Calculate final time.
                        self.final_time = time.time()
                        print("Final time: ", self.final_time)

                        # Calculate sentence time and converting it into seconds.
                        sentenceTime = self.final_time - initialTime
                        hours, rem = divmod(sentenceTime, 3600)
                        minutes, seconds = divmod(rem, 60)	

                        # Print out the number of blinks performed during sentence and the
                        # duration of it in seconds.
                        print("Blinks while talking ", self.blinks_while_talking)
                        print("Total seconds talking ", seconds)

                        # Print out the number of blink while talking per minute, floor
                        # rounded.
                        self.total_blinks_while_talking = self.blinks_while_talking * floor(60 / seconds)
                        print("Total blinks while talking per minute ", self.total_blinks_while_talking)

                        # Reset variables using during this process.
                        self.blinks_while_talking = 0
                        self.final_time = 0
                        initialTime = 0
                        sentenceTime = 0
                        self.is_talking = False
                        self.has_started_talking = False		

		        # draw the total number of blinks on the frame along with
		        # the computed eye aspect ratio for the frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "MAR: {:.2f}".format(MAR), (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # TODO: Check if they are really necessary on the next run.

                if self.total_blinks_while_talking > 0 and self.total_blinks_while_talking <= self.total_baseline_blinks:
                    cv2.putText(frame, "Lie", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.total_blinks_while_talking > 0 and self.total_blinks_while_talking >= self.total_baseline_blinks:
                    cv2.putText(frame, "Truth", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.counter_mouth >= MOUTH_AR_CONSEC_FRAMES and self.is_talking:
                    cv2.putText(frame, "Talking", (0, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if  self.counter_mouth < MOUTH_AR_CONSEC_FRAMES and self.is_talking == False:
                    cv2.putText(frame, "Not Talking", (0, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Mouth counter: {:.2f}".format(self.counter_mouth), (0, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.putText(frame, "Talking blinks: {:.2f}".format(self.total_blinks_while_talking), (0, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
	        # show the frame
            if (show_frames == True):
                cv2.imshow("Lie Detector", frame)

            key = cv2.waitKey(1) & 0xFF
 
	        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        self.is_program_ended = True

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

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


    def mouth_aspect_ratio(self, mouth):
        '''
        Mouth Aspect Ratio (MAR) calculation function
        '''
        # Instead of eye, mouth has 3 sets of vertical landmarks coordinates.
        innerA = dist.euclidean(mouth[13], mouth[19])
        innerB = dist.euclidean(mouth[14], mouth[18])
        innerC = dist.euclidean(mouth[15], mouth[17])

        # The sum of vertical euclidean distances is divided by three since
        # instead of
        # two since it must correspond to the cardinality of vertical
        # occurencies.
        meanInner = (innerA + innerB + innerC) / 3

        # Horizontal mouth euclidean distance.
        innerMouthBoundaries = dist.euclidean(mouth[12], mouth[16])

        # Compute Mouth Aspect Ratio (MAR)
        mouth = (meanInner / (3.0 * innerMouthBoundaries))

        return mouth


    def lie_analysis(self):
        '''
        '''
        audio_video_section_dir = "./data/chunks/{}".format(self.raw_file_name)
        if not os.path.isdir(audio_video_section_dir):
            os.mkdir(audio_video_section_dir)
        
        # TODO
        # creare file inizile per calcolare le costanti dell'occhio

        for t in self.tokens:
            data_row = []
            data_row.append("Token starts at {0} and ends at {1}".format(t[1] / 100, t[2] / 100))
            data = b''.join(t[0])

            video_section_path = "{}/{}-{}.mp4".format(audio_video_section_dir, t[1] / 100, t[2] / 100)
            if not os.path.exists(video_section_path):
                self.split_video_from_path(self.video_filename_path, "{}".format(t[1] / 100), "{}".format((t[2] - t[1]) / 100), video_section_path)

            self.eye_blink_detector(video_section_path, True)
            data_row.append("number eye blink {}".format(self.total))
            self.total = 0

            audio_section_path = "{}/{}-{}.mp3".format(audio_video_section_dir, t[1] / 100, t[2] / 100)
            if not os.path.exists(audio_section_path):
                self.split_audio_from_path(self.audio_filename_path, "{}".format(t[1] / 100), "{}".format((t[2] - t[1]) / 100), audio_section_path)

            y, sr = librosa.load(audio_section_path)
            pitches, magnitudes = librosa.piptrack(y, sr)
            pitches_tuning = librosa.pitch_tuning(pitches)
            data_row.append("Pitches: {}".format(pitches_tuning))

            self.output_data.append(data_row)

        return self.total_blinks_while_talking, self.total_baseline_blinks


    def split_audio_from_video(self):
        '''
        Split audio from video
        '''
        self.audio_filename_path = self.file_dir + '\\' + self.raw_file_name + '.wav'
        if not os.path.exists(self.audio_filename_path):
            subprocess.call(['ffmpeg', '-i', self.video_filename_path, '-codec:a', 'pcm_s16le', '-ac', '1', self.audio_filename_path])


    def split_video_from_path(self, input_file_name, start_time, end_time, output_file_name):
        '''
        Cut video chunk
        '''
        subprocess.call(['ffmpeg', '-i', input_file_name, "-ss", start_time, "-t", end_time, "-sn", output_file_name])


    def split_audio_from_path(self, input_file_name, start_time, end_time, output_file_name):
        '''
        Cut audio chunk
        '''
        subprocess.call(['ffmpeg', '-i', input_file_name, "-ss", start_time, "-t", end_time, "-sn", output_file_name])


    def baseline_blinks_per_minute(self):
        '''
        Function which calculates the number of baseline blinks over a period of 10 seconds, then a 
        multiplication is performed in order to obtain a number of baseline blinks per minute used 
        for future comparisons.
        '''
        while True:
            ticks = 0

            # Waiting 10 seconds before computing.
            while ticks <= self.maxBaselineLengthPeriod: 
                if self.is_program_ended == False:
                    return

                ticks += 1
                time.sleep(1.0)

		    # If it's the first time the algorithm is executed..
            if self.total_baseline_blinks == 0:
                self.total_baseline_blinks = (self.baseline_blinks) * 6 # 6 because the baseline period is 10 seconds long
                # ASK: Should we modify that 6 in order to use a value calculated on a specific proportion?

		    # Else, an average calculation is performed with the previous total baseline blinks value.
            else:
                self.total_baseline_blinks = (((self.baseline_blinks)* 6 + self.total_baseline_blinks) / 2)

		    # Print out the result.
            print("Baseline blinks per minute: ", self.total_baseline_blinks)
            self.baseline_blinks = 0
            ticks = 0