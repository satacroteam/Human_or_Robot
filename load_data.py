"""
Data Loader
"""
import pandas as pd


class Load(object):

    def __init__(self):
        self.bids = None
        self.train = None
        self.test = None

        self.bids_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\bids.csv"
        self.train_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\train.csv"
        self.test_path = r"C:\Users\Robinet Florian\Desktop\facebook_auction-master\test.csv"

        self.load_initial_data()

    def load_initial_data(self):
        self.bids = pd.read_csv(self.bids_path, sep=",")
        self.bids.fillna('-', inplace=True)
        self.train = pd.read_csv(self.train_path, sep=",")
        self.test = pd.read_csv(self.test_path, sep=",")