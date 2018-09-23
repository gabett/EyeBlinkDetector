import SvmDetector
import Classifier
import NaifDetector

if __name__ == '__main__':

    classifier = Classifier.Classifier()
    #classifier.train300VW("D:\\Datasets\\300VW\\TS.txt")
    #classifier.testEyeblink8()
    #classifier = Classifier.Classifier()
    #classifier.trainEyeblink8()
    #classifier.test300VW("D:\\Datasets\\300VW\\TS.txt")
    Detector = SvmDetector.SvmDetector()
    Detector.Start("D:\\Google Drive\\University\\Data Science\\Courses\\COB\\Videos\\Ale\\BaselineAle.mp4", True)
    #naif = NaifDetector.NaifDetector()
    #naif.Start("D:\\Google Drive\\University\\Data Science\\Courses\\COB\\Videos\\Ale\\BaselineAle.mp4")