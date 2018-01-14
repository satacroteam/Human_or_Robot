"""
Data Loader
"""
import pandas as pd


class Load(object):

    def __init__(self):
        self.bids = None
        self.train = None
        self.test = None
        self.train_test_concat = None

        self.bids_path = "data/bids.csv"
        self.train_path = "data/train.csv"
        self.test_path = "data/test.csv"

        self.load_initial_data()

    def load_initial_data(self):
        """
        Load initial data for the model
        """
        self.bids = pd.read_csv(self.bids_path, sep=",")
        self.bids.fillna(0, inplace=True)
        self.train = pd.read_csv(self.train_path, sep=",")
        self.test = pd.read_csv(self.test_path, sep=",")
        # self.test['outcome'] = -1.0
        self.train_test_concat = pd.concat((self.train, self.test))
