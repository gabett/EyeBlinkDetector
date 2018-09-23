from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import numpy as np

class Classifier:

    def __init__(self):
        self.trainingFeatures = []
        self.trainingLabels = []
        self.SVC = LinearSVC()


    def train300VW(self, filePath):
        numberOfNegatives = 0
        numberOfPositives = 0
        self.trainingLabels = []

        with open(filePath) as f:
            lines = f.readlines()

            for line in lines:
                items = line.rstrip('\n').split("   ")
                features = [float(x) for x in items[:13]]
                label = int(items[14].strip())

                if label == 1:
                    numberOfPositives += 1
                    self.trainingLabels.append(1)

                elif label == 0:
                    numberOfNegatives += 1
                    self.trainingLabels.append(0)

                self.trainingFeatures = np.append(self.trainingFeatures, np.asarray(features))
                self.trainingFeatures = np.reshape(self.trainingFeatures, (-1, 13))

            self.SVC.fit(self.trainingFeatures, self.trainingLabels)

    def trainEyeblink8(self):
        numberOfNegatives = 0
        numberOfPositives = 0

        filePath = ".\\Training Sets\\Negative_EarBlinks\\negative_ear"
        fileIndices = ['', '2', '3', '4', '5', '6', '7', '8', '9'] 

        for i in fileIndices:
            with open(filePath + i + '.txt') as f:
                
                lines = f.readlines()

                for line in lines:
                    X = line.rstrip('\n').split(" ")
                    X = [float(x) for x in X if x != str('')]
                    self.trainingFeatures = np.append(self.trainingFeatures, np.asarray(X))
                    numberOfNegatives +=1
       
        self.trainingLabels = np.zeros(numberOfNegatives)

        filePath = ".\\Training Sets\\Positive_EarBlinks\\positive_ear"
        
        numberOfPositives = 0
        
        for i in fileIndices:
            with open(filePath + i + '.txt') as f:

                lines = f.readlines()
        
                for line in lines:
                    X = line.rstrip('\n').split(" ")
                    X = [float(x) for x in X if x != str('')]
                    self.trainingFeatures = np.append(self.trainingFeatures, np.asarray(X))
                    numberOfPositives += 1
            
        self.trainingLabels = np.append(self.trainingLabels, np.ones(numberOfPositives))
        self.trainingFeatures = np.reshape(self.trainingFeatures, (-1, 13))

        self.SVC.fit(self.trainingFeatures, self.trainingLabels)
        return

    def classify(self, featureRow):
        # Make sure a numpy array is being used.
        arrayToClassify = np.array(featureRow) 
        outcome = self.SVC.predict(np.reshape(arrayToClassify, (1, -1)))
        return outcome

    def getScore(self, testingFeatures, testingLabels):
        # Make sure a Nx13 matrix is being used.
        testFeatures = np.reshape(testFeatures, (-1, 1))
        score = clf.score(testFeatures, y)
        return score

    def test300VW(self, filePath):
        numberOfNegatives = 0
        numberOfPositives = 0
        testFeatures = []
        y = []

        with open(filePath) as f:
            lines = f.readlines()

            for line in lines:
                items = line.rstrip('\n').split("   ")
                features = [float(x) for x in items[:13]]
                label = int(items[14].strip())

                if label == 1:
                    numberOfPositives += 1
                    y.append(1)

                elif label == 0:
                    numberOfNegatives += 1
                    y.append(0)

                testFeatures = np.append(testFeatures, np.asarray(features))
                testFeatures = np.reshape(testFeatures, (-1, 13))

            print("Accuracy: ", self.SVC.score(testFeatures, y))
        return

    def testEyeblink8(self):
        filePath = ".\\Test Sets\\negative_ear.txt"
        numberOfNegatives = 0

        testFeatures = []
        with open(filePath) as f:
            lines = f.readlines()

            for line in lines:
                X = line.rstrip('\n').split(" ")
                X = [float(x) for x in X if x != str('')]
                testFeatures = np.append(testFeatures, np.asarray(X))
                numberOfNegatives +=1

        y = np.zeros(numberOfNegatives)

        filePath = ".\\Test Sets\\positive_ear.txt"

        numberOfPositives = 0

        with open(filePath) as f:
            lines = f.readlines()

            for line in lines:
                X = line.rstrip('\n').split(" ")
                X = [float(x) for x in X if x != str('')]
                testFeatures = np.append(testFeatures, np.asarray(X))
                numberOfPositives += 1

        y = np.append(y, np.ones(numberOfPositives))
        testFeatures = np.reshape(testFeatures, (-1, 13))

        print("Accuracy: ", self.SVC.score(testFeatures, y))
        return