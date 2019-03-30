import sys
import pandas as pd
import numpy as np
from PIL import Image
from network import Network


def readTrainingData():
    data = pd.read_csv("./data/train.csv")
    features = data.values[:, 1:]
    outputs = [vectorized_result(x) for x in data.values[:,0]]
    return [(feature, output) for feature, output in zip(features, outputs)]

def vectorized_result(x):
    res = np.zeros((10, 1))
    res[x] = 1
    return res

# def readTestData():
#     data = pd.read_csv("./data/train.csv")
#     features = data.values[:, 1:]
#     outputs = data.values[:, 0]
#     return [(feature, output) for feature, output in zip(features, outputs)]


def createImage(pixelValues):
    imgSize = 28
    img = Image.new("L", (imgSize, imgSize))
    img.putdata(pixelValues)
    return img


def normalizeData(data):
    """``data`` is a list of tuples (features, output)"""
    features = np.asarray([x[0].astype(float) for x in data])
    outputs = [x[1] for x in data]
    maxValues = np.amax(features, axis=1).reshape(len(features), 1)
    features[:, ] /= maxValues
    return [(x.reshape(len(x), 1), y) for x, y in zip(features, outputs)]


def main():
    training_data = readTrainingData()
    training_data = normalizeData(training_data)
    test_data = training_data[0:1000]

    net = Network([784, 10, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


main()
