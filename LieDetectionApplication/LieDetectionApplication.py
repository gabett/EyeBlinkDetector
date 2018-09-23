import liedetection
import lieclassifier

#lie_classifier = lieclassifier.LieClassifier().InitializeClassifier()
lie_detection = liedetection.LieDetection(".\\VeritaAnna1.mp4")
lie_detection.tokenize_audio()
total_blinks_while_talking, total_baseline_blinks = lie_detection.lie_analysis()
#lie_detection.print_data()
#total_blinks_while_talking = 18
#lie_classifier.TestSentence(total_blinks_while_talking, 1)

