import pickle
import pandas as pd
import xgboost as xgb

import numpy as np


class Model(object):
    def __init__(self):
        self.train_data = self.load('train_data_set.pkl')
        self.train_answer = self.load('train_answer.pkl')
        self.test_data = [test_id for test_id in pd.read_csv("test.csv", sep=',')['bidder_id']]
        self.model = None

    @staticmethod
    def load(pickle_name):
        with open(pickle_name, 'rb') as data_file:
            data = pickle.load(data_file)
        return data

    def train(self):
        y_mean = np.mean([int(value) for value in self.train_answer.values()])

        xgb_params = {
            'n_trees': 500,
            'eta': 0.005,
            'max_depth': 4,
            'subsample': 0.90,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'base_score': y_mean,  # base prediction = mean(target)
            'silent': 1
        }

        # NOTE: Make sure that the class is labeled 'class' in the data file
        train = self.train_data[['nb_unique_ip', 'low_freq_ip', 'high_freq_ip', 'std_freq_ip',
                                 'nb_unique_device', 'low_freq_device', 'high_freq_device', 'std_freq_device',
                                 'nb_unique_url', 'low_freq_url', 'high_freq_url', 'std_freq_url',
                                 'nb_unique_auction', 'low_freq_auction', 'high_freq_auction', 'std_freq_auction']]

        train = pd.DataFrame(data=train, index=self.train_answer.keys()).values
        answer = [int(value) for value in self.train_answer.values()]

        d_matrix_train = xgb.DMatrix(train, answer)

        num_boost_rounds = 1250

        self.model = xgb.train(dict(xgb_params, silent=0), d_matrix_train, num_boost_round=num_boost_rounds)

    def test(self):
        test = self.train_data[['nb_unique_ip', 'low_freq_ip', 'high_freq_ip', 'std_freq_ip',
                                 'nb_unique_device', 'low_freq_device', 'high_freq_device', 'std_freq_device',
                                 'nb_unique_url', 'low_freq_url', 'high_freq_url', 'std_freq_url',
                                 'nb_unique_auction', 'low_freq_auction', 'high_freq_auction', 'std_freq_auction']]

        test = pd.DataFrame(data=test, index=self.test_data).values
        dtest = xgb.DMatrix(test)
        y_pred = self.model.predict(dtest)
        test_answer = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(y_pred)], ignore_index=True, names=["bidder_id", "prediction"], axis=1)
        print(test_answer)
        pd.DataFrame(test_answer).to_csv('08_01_18.csv', index=False)

model = Model()
model.train()
model.test()