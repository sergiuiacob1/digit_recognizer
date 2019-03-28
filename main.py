import pandas as pd


def read_data():
    data = pd.read_csv("./data/train.csv")
    data.to_csv()

read_data()
