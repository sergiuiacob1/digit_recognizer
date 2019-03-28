import pandas as pd
from PIL import Image


def readData():
    data = pd.read_csv("./data/small_train.csv")
    return data


def createImage(pixelValues):
    imgSize = 28
    img = Image.new("L", (imgSize, imgSize))
    img.putdata(pixelValues)
    return img


data = readData()
img = createImage(data.head(1).values[0])
