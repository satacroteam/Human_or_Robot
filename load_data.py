"""
Data Loader
"""
import pandas as pd


class Load(object):

    def __init__(self):
        self.x_train = None
        self.y_train = None

        self.bids_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\bids.csv"
        self.train_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\train.csv"
        self.test_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\test.csv"

    def load_initial_data(self):
        bids = pd.read_csv(self.bids_path, sep=",")
        train = pd.read_csv(self.train_path, sep=",")
        test = pd.read_csv(self.test_path, sep=",")

        return bids, train, test