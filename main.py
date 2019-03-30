import sys
import pandas as pd
import numpy as np
from PIL import Image
from network import Network


def readTrainingData():
    data = pd.read_csv("./data/small_train.csv")
    features = data.values[:, 1:]
    outputs = data.values[:, 0]
    return [(feature, output) for feature, output in zip(features, outputs)]


# def readTestData():
#     data = pd.read_csv("./data/test.csv")
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
    maxValues = np.amax(features, axis=1, keepdims=True)
    features[:, ] /= maxValues
    return [(x.reshape(len(x), 1), y) for x, y in zip(features, outputs)]


def main():
    data = readTrainingData()
    data = normalizeData(data)
    training_data = data[1:len(data) - 10000]
    validation_data = data[-10000:]
    net = Network([784, 10, 10])
    net.SGD(training_data, 1, 10, 3.0, test_data=validation_data)


main()
