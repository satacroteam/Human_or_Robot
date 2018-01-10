"""
Extract features to build X
"""
import os
import pickle
import numpy as np
import pandas as pd

from src.load_data import Load


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

        nb_unique = np.float(counts.count())
        high_freq = np.float(counts[0] / n)
        low_freq = np.float(counts[-1] / n)
        arg_max = counts.index[0]
        std_freq = np.float(np.std(counts / n))

        return nb_unique, low_freq, high_freq, std_freq, arg_max

    @staticmethod
    def compute_stats_for_time_series(series):

        bid_nb = np.float(series.shape[0])
        min_time = np.float(series.min())
        max_time = np.float(series.max())
        range_time = np.float(max_time - min_time)

        time_interval = series[1:].as_matrix() - series[:-1].as_matrix()

        if len(time_interval) < 1:
            time_interval = np.array([0])

        min_time_interval = np.float(np.min(time_interval))
        max_time_interval = np.float(np.max(time_interval))

        mean_time_interval = np.float(np.mean(time_interval))
        std_time_interval = np.float(np.std(time_interval))

        time_interval_25 = np.float(np.percentile(time_interval, 25))
        time_interval_50 = np.float(np.percentile(time_interval, 50))
        time_interval_75 = np.float(np.percentile(time_interval, 75))

        return bid_nb, min_time, max_time, range_time, min_time_interval, max_time_interval, mean_time_interval, \
               std_time_interval, time_interval_25, time_interval_50, time_interval_75

    # Compute stats about a numerical column of table, with stats on sub-groups of this column (auctions in our case).
    def compute_stats_for_time_series_with_group_by(self, table, column, group_by):

        # get series and groups
        series = table[column]
        groups = table.groupby(group_by)

        # global stats
        bid_nb, min_time, max_time, range_time, min_time_interval, max_time_interval, mean_time_interval, \
        std_time_interval, time_interval_25, time_interval_50, time_interval_75 = self.compute_stats_for_time_series(
            series)

        # stats by group
        auction_data = []
        for _, group in groups:
            auction_bid_nb, auction_min_time, auction_max_time, auction_range_time, auction_min_time_interval, \
            auction_max_time_interval, auction_mean_time_interval, auction_std_time_interval, auction_time_interval_25, \
            auction_time_interval_50, auction_time_interval_75 = self.compute_stats_for_time_series(group[column])

            auction_data.append([
                auction_bid_nb,
                auction_range_time,
                auction_min_time_interval,
                auction_max_time_interval,
                auction_mean_time_interval,
                auction_std_time_interval]
            )

        auction_data = np.array(auction_data)

        mean_of_auction_bid_nb = np.float(np.mean(auction_data[:, 0]))
        std_of_auction_bid_nb = np.float(np.std(auction_data[:, 0]))
        mean_of_auction_range_time = np.float(np.mean(auction_data[:, 1]))
        std_of_auction_range_time = np.float(np.std(auction_data[:, 1]))
        min_of_auction_min_time_interval = np.float(np.min(auction_data[:, 2]))
        mean_of_auction_min_time_interval = np.float(np.mean(auction_data[:, 2]))
        max_auction_max_time_interval = np.float(np.max(auction_data[:, 3]))
        mean_auction_max_time_interval = np.float(np.mean(auction_data[:, 3]))
        mean_auction_mean_time_interval = np.float(np.mean(auction_data[:, 4]))
        std_auction_mean_time_interval = np.float(np.std(auction_data[:, 4]))
        mean_auction_std_time_interval = np.float(np.mean(auction_data[:, 5]))

        return bid_nb, min_time, max_time, range_time, min_time_interval, max_time_interval, mean_time_interval, \
               std_time_interval, time_interval_25, time_interval_50, time_interval_75, mean_of_auction_bid_nb, \
               std_of_auction_bid_nb, mean_of_auction_range_time, std_of_auction_range_time, \
               min_of_auction_min_time_interval, mean_of_auction_min_time_interval, max_auction_max_time_interval, \
               mean_auction_max_time_interval, mean_auction_mean_time_interval, std_auction_mean_time_interval, \
               mean_auction_std_time_interval

    def extract(self):
        if not os.path.isfile('pickle/train_ids.pkl'):
            print("Extract IDs")
            for bidder_id, group in self.load_data.bids.groupby('bidder_id'):
                self.train_ids.append(bidder_id)
            with open('pickle/train_ids.pkl', 'wb') as train_ids_file:
                pickle.dump(self.train_ids, train_ids_file)
        else:
            print("IDs already created")
            # print("Load IDs")
            # with open('pickle/train_ids.pkl', 'rb') as train_ids_file:
            # self.train_ids = pickle.load(train_ids_file)

        if not os.path.isfile('pickle/train.pkl'):
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

                bid_nb, min_time, max_time, range_time, min_time_interval, max_time_interval, mean_time_interval, \
                std_time_interval, time_interval_25, time_interval_50, time_interval_75, mean_of_auction_bid_nb, \
                std_of_auction_bid_nb, mean_of_auction_range_time, std_of_auction_range_time, \
                min_of_auction_min_time_interval, mean_of_auction_min_time_interval, max_auction_max_time_interval, \
                mean_auction_max_time_interval, mean_auction_mean_time_interval, std_auction_mean_time_interval, \
                mean_auction_std_time_interval = self.compute_stats_for_time_series_with_group_by(group, 'time', 'auction')

                statistical_results = [
                    nb_unique_ip, low_freq_ip, high_freq_ip, std_freq_ip, arg_max_ip,
                    nb_unique_device, low_freq_device, high_freq_device, std_freq_device, arg_max_device,
                    nb_unique_merchandise, low_freq_merchandise, high_freq_merchandise, std_freq_merchandise,
                    arg_max_merchandise,
                    nb_unique_country, low_freq_country, high_freq_country, std_freq_country, arg_max_country,
                    nb_unique_url, low_freq_url, high_freq_url, std_freq_url, arg_max_url,
                    nb_unique_auction, low_freq_auction, high_freq_auction, std_freq_auction, arg_max_auction,
                    bid_nb, min_time, max_time, range_time, min_time_interval, max_time_interval, mean_time_interval,
                    std_time_interval, time_interval_25, time_interval_50, time_interval_75, mean_of_auction_bid_nb,
                    std_of_auction_bid_nb, mean_of_auction_range_time, std_of_auction_range_time,
                    min_of_auction_min_time_interval, mean_of_auction_min_time_interval, max_auction_max_time_interval,
                    mean_auction_max_time_interval, mean_auction_mean_time_interval, std_auction_mean_time_interval,
                    mean_auction_std_time_interval

                ]

                self.train.append(statistical_results)

            with open('pickle/train.pkl', 'wb') as train_file:
                pickle.dump(self.train, train_file)
        else:
            print("Train already created")
            # print("Load features")
            # with open('pickle/train.pkl', 'rb') as train_file:
            # self.train = pickle.load(train_file)

        if not os.path.isfile('pickle/train_data_set.pkl'):
            print("Building data set with the features")
            column_names = [
                'nb_unique_ip', 'low_freq_ip', 'high_freq_ip', 'std_freq_ip', 'arg_max_ip',
                'nb_unique_device', 'low_freq_device', 'high_freq_device', 'std_freq_device', 'arg_max_device',
                'nb_unique_merchandise', 'low_freq_merchandise', 'high_freq_merchandise', 'std_freq_merchandise',
                'arg_max_merchandise',
                'nb_unique_country', 'low_freq_country', 'high_freq_country', 'std_freq_country', 'arg_max_country',
                'nb_unique_url', 'low_freq_url', 'high_freq_url', 'std_freq_url', 'arg_max_url',
                'nb_unique_auction', 'low_freq_auction', 'high_freq_auction', 'std_freq_auction', 'arg_max_auction',
                "bid_nb", "min_time", "max_time", "range_time", "min_time_interval", "max_time_interval",
                "mean_time_interval", "std_time_interval", "time_interval_25", "time_interval_50", "time_interval_75",
                "mean_of_auction_bid_nb", "std_of_auction_bid_nb", "mean_of_auction_range_time",
                "std_of_auction_range_time", "min_of_auction_min_time_interval", "mean_of_auction_min_time_interval",
                "max_auction_max_time_interval", "mean_auction_max_time_interval", "mean_auction_mean_time_interval",
                "std_auction_mean_time_interval", "mean_auction_std_time_interval"
            ]

            with open('pickle/train_ids.pkl', 'rb') as train_ids_file:
                self.train_ids = pickle.load(train_ids_file)

            with open('pickle/train.pkl', 'rb') as train_file:
                self.train = pickle.load(train_file)

            data_set = pd.DataFrame(self.train, index=self.train_ids, columns=column_names)
            data_set.fillna(0.0, inplace=True)

            with open('pickle/train_data_set.pkl', 'wb') as train_data_set_file:
                pickle.dump(data_set, train_data_set_file)
        else:
            print("Train data set already created")

    def build_answer(self):
        if not os.path.isfile('pickle/train_answer.pkl'):
            print("Building answers ids data")
            dict_ids_outcome = {}
            with open('pickle/train_ids.pkl', 'rb') as train_ids_file:
                self.train_ids = pickle.load(train_ids_file)
                for train_bidder_id, outcome in zip(self.load_data.train['bidder_id'], self.load_data.train['outcome']):
                    print(train_bidder_id, outcome)
                    dict_ids_outcome[train_bidder_id] = outcome

            self.train_answer = dict_ids_outcome
            print(self.train_answer)

            with open('pickle/train_answer.pkl', 'wb') as train_answer_file:
                pickle.dump(self.train_answer, train_answer_file)
        else:
            print("Train answer data set already created")
