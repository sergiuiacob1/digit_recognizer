import _pickle as pickle
import pandas as pd
import numpy as np
from PIL import Image
from network import Network


def readTrainingData():
    data = pd.read_csv("./data/small_train.csv")
    features = data.values[:, 1:]
    outputs = [vectorized_result(x) for x in data.values[:, 0]]
    return [(feature, output) for feature, output in zip(features, outputs)]


def vectorized_result(x):
    res = np.zeros((10, 1))
    res[x] = 1
    return res


def readTestData():
    data = pd.read_csv("./data/test.csv")
    return data.values


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


def normalizeTestData(testData):
    features = np.asarray([x.astype(float) for x in testData])
    maxValues = np.amax(features, axis=1).reshape(len(features), 1)
    features[:, ] /= maxValues
    return [feature.reshape(28*28, 1) for feature in features]


def trainNetwork(net, training_data, test_data):
    (weights, biases) = net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
    serialized = pickle.dumps((weights, biases), protocol=0)
    f = open("parameters.txt", "wb")
    f.write(serialized)
    f.close()


def savePredictions(predictions):
    f = open("predictions.csv", "w")
    f.write("ImageId,Label\n")
    for x in range(len(predictions)):
        f.write("{},{}\n".format(x + 1, predictions[x]))


def main():
    net = Network([784, 100, 10])

    # training_data = readTrainingData()
    # training_data = normalizeData(training_data)
    # trainNetwork(net, training_data, test_data)

    test_data = readTestData()
    test_data = normalizeTestData(test_data)

    predictions = net.predict(test_data)
    savePredictions(predictions)
    # outputs = [x[1] for x in test_data]
    # print(sum([prediction == np.argmax(output)
    #            for prediction, output in zip(predictions, outputs)]))


main()
