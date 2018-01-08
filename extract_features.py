"""
Extract features to build X
"""
import os
import pickle
import numpy as np
import pandas as pd

from load_data import Load


class Extract(object):

    def __init__(self):
        self.load_data = Load()
        self.train_ids = []
        self.train = []
        self.train_data_set = None
        self.train_answer = pd.DataFrame()

    @staticmethod
    def compute_stats_by_categories(series):
        n = float(series.shape[0])
        counts = series.value_counts()

        nb_unique = counts.count()
        high_freq = counts[0] / n
        low_freq = counts[-1] / n
        arg_max = counts.index[0]
        std_freq = np.std(counts / n)

        return nb_unique, low_freq, high_freq, std_freq, arg_max

    def extract(self):
        if not os.path.isfile('train_ids.pkl'):
            print("Extract IDs")
            for bidder_id, group in self.load_data.bids.groupby('bidder_id'):
                self.train_ids.append(bidder_id)
            with open('train_ids.pkl', 'wb') as train_ids_file:
                pickle.dump(self.train_ids, train_ids_file)
        else:
            print("IDs already created")
            # print("Load IDs")
            # with open('train_ids.pkl', 'rb') as train_ids_file:
                # self.train_ids = pickle.load(train_ids_file)

        if not os.path.isfile('train.pkl'):
            print("Extract features")
            for bidder_id, group in self.load_data.bids.groupby('bidder_id'):
                print(bidder_id)
                nb_unique_ip, low_freq_ip, high_freq_ip, std_freq_ip, arg_max_ip = \
                    self.compute_stats_by_categories(group.ip)
                nb_unique_device, low_freq_device, high_freq_device, std_freq_device, arg_max_device = \
                    self.compute_stats_by_categories(group.device)
                nb_unique_merchandise, low_freq_merchandise, high_freq_merchandise, std_freq_merchandise, arg_max_merchandise = \
                    self.compute_stats_by_categories(group.merchandise)
                nb_unique_country, low_freq_country, high_freq_country, std_freq_country, arg_max_country = \
                    self.compute_stats_by_categories(group.country)
                nb_unique_url, low_freq_url, high_freq_url, std_freq_url, arg_max_url = \
                    self.compute_stats_by_categories(group.url)
                nb_unique_auction, low_freq_auction, high_freq_auction, std_freq_auction, arg_max_auction = \
                    self.compute_stats_by_categories(group.auction)

                statistical_results = [
                    nb_unique_ip, low_freq_ip, high_freq_ip, std_freq_ip, arg_max_ip,
                    nb_unique_device, low_freq_device, high_freq_device, std_freq_device, arg_max_device,
                    nb_unique_merchandise, low_freq_merchandise, high_freq_merchandise, std_freq_merchandise, arg_max_merchandise,
                    nb_unique_country, low_freq_country, high_freq_country, std_freq_country, arg_max_country,
                    nb_unique_url, low_freq_url, high_freq_url, std_freq_url, arg_max_url,
                    nb_unique_auction, low_freq_auction, high_freq_auction, std_freq_auction, arg_max_auction

                ]

                self.train.append(statistical_results)

            with open('train.pkl', 'wb') as train_file:
                pickle.dump(self.train, train_file)
        else:
            print("Train already created")
            # print("Load features")
            # with open('train.pkl', 'rb') as train_file:
                # self.train = pickle.load(train_file)

        if not os.path.isfile('train_data_set.pkl'):
            print("Building data set with the features")
            column_names = [
                'nb_unique_ip', 'low_freq_ip', 'high_freq_ip', 'std_freq_ip', 'arg_max_ip',  # 'IP',
                'nb_unique_device', 'low_freq_device', 'high_freq_device', 'std_freq_device', 'arg_max_device',  # 'device',
                'nb_unique_merchandise', 'low_freq_merchandise', 'high_freq_merchandise', 'std_freq_merchandise', 'arg_max_merchandise',  # merchandise ,
                'nb_unique_country', 'low_freq_country', 'high_freq_country', 'std_freq_country', 'arg_max_country',  # 'country',
                'nb_unique_url', 'low_freq_url', 'high_freq_url', 'std_freq_url', 'arg_max_url',  # 'url',
                'nb_unique_auction', 'low_freq_auction', 'high_freq_auction', 'std_freq_auction', 'arg_max_auction',  # 'auction' ,
                ]

            with open('train_ids.pkl', 'rb') as train_ids_file:
                self.train_ids = pickle.load(train_ids_file)

            with open('train.pkl', 'rb') as train_file:
                self.train = pickle.load(train_file)

            data_set = pd.DataFrame(self.train, index=self.train_ids, columns=column_names)
            data_set.fillna(0.0, inplace=True)

            with open('train_data_set.pkl', 'wb') as train_data_set_file:
                pickle.dump(data_set, train_data_set_file)
        else:
            print("Train data set already created")

    def build_answer(self):
        if not os.path.isfile('train_data_set.pkl'):
            print("Building answers ids data")
            dict_ids_outcome = {}
            with open('train_ids.pkl', 'rb') as train_ids_file:
                self.train_ids = pickle.load(train_ids_file)
                for train_bidder_id, outcome in zip(self.load_data.train['bidder_id'], self.load_data.train['outcome']):
                    print(train_bidder_id, outcome)
                    dict_ids_outcome[train_bidder_id] = outcome

            self.train_answer = dict_ids_outcome
            print(self.train_answer)

            with open('train_answer.pkl', 'wb') as train_answer_file:
                pickle.dump(self.train_answer, train_answer_file)
        else:
            print("Train answer data set already created")