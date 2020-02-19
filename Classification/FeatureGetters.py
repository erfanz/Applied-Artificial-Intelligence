import os
import numpy as np
import random
import cv2

class EmailFeatureGetter():
    def __init__(self, maxWords, downSamplingRate, trainingSetFileLoc="data/train_small.txt"):
        self.maxWords = maxWords
        self.downSamplingRate = downSamplingRate

        f = open(trainingSetFileLoc, 'r')
        data = f.readlines()
        f.close()
        ham = [data[i].split()[:-1] for i in range(len(data)) if data[i].endswith("HAM\n") or data[i].endswith("HAM\r") or data[i].endswith("HAM\r\n")]
        spam = [data[i].split()[:-1] for i in range(len(data)) if data[i].endswith("SPAM\n") or data[i].endswith("SPAM\r") or data[i].endswith("SPAM\r\n")]
        
        self.wordToBitMaskIndex = self.getWordToBitMaskIndexDict(ham, spam)
        self.bitMaskLength = len(self.wordToBitMaskIndex)

    def getFeaturizedEmail(self, email):
        toReturn = np.zeros(self.bitMaskLength/self.downSamplingRate, np.int32)
        for word in email:
            if word in self.wordToBitMaskIndex:
                bitMaskIndex = self.wordToBitMaskIndex[word]/self.downSamplingRate
                toReturn[bitMaskIndex] += 1
        return toReturn

    def getWordToBitMaskIndexDict(self, ham, spam):
        allWords = {}
        index = 0
        random.seed(5)

        totalEmails = ham + spam
        random.shuffle(totalEmails)
        for email in totalEmails:
            for word in email:
                if not word in allWords:
                    allWords[word] = index
                    index += 1
                    if index >= self.maxWords:
                        return allWords 
        return allWords

    def getFeaturesAndLabelsFromFileLoc(self, fileLoc):
        f = open(fileLoc, 'r')
        data = f.readlines()
        f.close()
        ham = [data[i].split()[:-1] for i in range(len(data)) if data[i].endswith("HAM\n") or data[i].endswith("HAM\r") or data[i].endswith("HAM\r\n")]
        spam = [data[i].split()[:-1] for i in range(len(data)) if data[i].endswith("SPAM\n") or data[i].endswith("SPAM\r") or data[i].endswith("SPAM\r\n")]

        y = [0 for i in range(len(ham))] + [1 for i in range(len(spam))]
        allEmails = ham + spam

        X = [self.getFeaturizedEmail(email) for email in allEmails]

        return X, y


class DoorFeatures:
    def __init__(self, root_dir):

        door_dir = root_dir + "/doors/"
        not_door_dir = root_dir + "/not_doors/"

        bin_avgs = dict() # dictionary to store average values for each bin
        door_hists = [] # array to store door histograms
        not_door_hists = [] # array to store non-door histograms
        X = []
        y = []

        # Sum bin values for door images
        for f in os.listdir(door_dir):
            img = cv2.imread(door_dir + f, cv2.CV_LOAD_IMAGE_COLOR)
            b_hist = cv2.calcHist([img], [0], None, [25], [0,256])
            g_hist = cv2.calcHist([img], [1], None, [25], [0,256])
            r_hist = cv2.calcHist([img], [2], None, [25], [0,256])
            hist = b_hist + g_hist + r_hist
            door_hists.append(hist)
            for b in range(len(hist)):
                if not b in bin_avgs:
                    bin_avgs[b] = 0
                bin_avgs[b] += hist[b][0] 

        # Sum bin values for non-door images
        for f in os.listdir(not_door_dir):
            img = cv2.imread(not_door_dir + f, cv2.CV_LOAD_IMAGE_COLOR)
            b_hist = cv2.calcHist([img], [0], None, [25], [0,256])
            g_hist = cv2.calcHist([img], [1], None, [25], [0,256])
            r_hist = cv2.calcHist([img], [2], None, [25], [0,256])
            hist = b_hist + g_hist + r_hist
            not_door_hists.append(hist)
            for b in range(len(hist)):
                if not b in bin_avgs:
                    bin_avgs[b] = 0
                bin_avgs[b] += hist[b][0] 

        # Calculate average bin values
        for b in range(len(bin_avgs.keys())):
            bin_avgs[b] = bin_avgs[b] / float(len(door_hists) + len(not_door_hists))

        # Generate average bitmasks and labels for every histogram
        for hist in door_hists:
            x = []
            for b in range(len(hist)):
                if hist[b] > bin_avgs[b]:
                    x.append(1)
                else:
                    x.append(0)
            X.append(x)
            y.append(1) # label as a door

        for hist in not_door_hists:
            x = []
            for b in range(len(hist)):
                if hist[b] > bin_avgs[b]:
                    x.append(1)
                else:
                    x.append(0)
            X.append(x)
            y.append(0) # label as not a door

        self.X = X
        self.y = y

    def getFeatures(self):
        return (self.X, self.y)

if __name__ == "__main__":
    eFGetter = EmailFeatureGetter()
    print eFGetter.getTrainingFeaturesAndLabels()
