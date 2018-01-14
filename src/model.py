"""
Study model
"""
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
from matplotlib import pyplot

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


class Model(object):
    def __init__(self):
        self.train_data = self.load('pickle/train_data_set.pkl')
        self.train_answer = self.load('pickle/train_answer.pkl')
        self.test_data = [test_id for test_id in pd.read_csv("data/test.csv", sep=',')['bidder_id']]
        self.mapper = None
        self.feature_selection_list = None
        self.model = None
        self.n_comp = 3
        self.pca = None
        self.t_svd = None

    @staticmethod
    def load(pickle_name):
        with open(pickle_name, 'rb') as data_file:
            data = pickle.load(data_file)
        return data

    def feature_engineering(self, data, answer=None):
        """
        Feature engineering for data (PCA, etc...)
        :param data: Data to fit_transform
        :param answer: Answer series for data
        :return: Data with new features
        """
        if answer:
            # PCA
            self.pca = PCA(n_components=self.n_comp, random_state=420)
            self.pca.fit(np.nan_to_num(data), answer)
            pca_results_train = self.pca.transform(np.nan_to_num(data))

            # tSVD
            '''self.t_svd = KernelPCA(n_components=self.n_comp, random_state=420)
            t_svd_results_train = self.t_svd.fit_transform((np.nan_to_num(data)))'''

            data = pd.DataFrame(data)

            for i in range(1, self.n_comp + 1):
                data['pca_' + str(i)] = pca_results_train[:, i - 1]
                '''data['t_svd_' + str(i)] = t_svd_results_train[:, i - 1]'''
        else:

            pca_results_test = self.pca.transform(np.nan_to_num(data))

            ''' t_svd_results_test = self.t_svd.transform(np.nan_to_num(data))'''

            data = pd.DataFrame(data)

            for i in range(1, self.n_comp + 1):
                data['pca_' + str(i)] = pca_results_test[:, i - 1]
                '''data['t_svd_' + str(i)] = t_svd_results_test[:, i - 1]'''

        return data

    def feature_selection(self, data):
        return data[
            [8514, 3699, 8523, 4403, 'pca_1', 8504, 6964, 8506, 2, 8511, 0, 4405, 4406, 4302, 'pca_2', 8516, 3, 8512, 3698,
             'pca_3', 8510, 8522, 2801, 4345, 6961, 3700, 8503, 8519, 4297, 8508, 8515, 4052, 8517, 4404, 6963, 8521,
             8524, 4304, 3701, 4301, 8520, 4298, 8509, 8505, 8513, 4303, 7807, 4296, 4397, 4300, 7819, 8507, 4342, 1,
             4090, 6644, 6962, 3964]

        ]

    def train(self):
        """
        Train the model
        """
        train = pd.DataFrame(data=self.train_data, index=self.train_answer.keys())
        answer = [int(value) for value in self.train_answer.values()]
        train = self.feature_engineering(train, answer)

        self.feature_selection_list = train.columns.tolist()

        train = pd.DataFrame(data=train, columns=self.feature_selection_list, index=self.train_answer.keys())

        train = self.feature_selection(train)

        print(len(train), len(answer))

        y_mean = np.mean([int(value) for value in self.train_answer.values()])
        num_boost_rounds = 1250

        xgb_params = {
            'n_trees': 9000,
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.89,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'base_score': y_mean,  # base prediction = mean(target)
            'silent': 1
        }

        d_matrix_train = xgb.DMatrix(train, answer)
        self.model = xgb.train(dict(xgb_params, silent=0), d_matrix_train, num_boost_round=num_boost_rounds)

        print(self.model.get_fscore().items())

        xgb.plot_importance(self.model)
        pyplot.show()

    def test(self):
        """
        Predict answer for test data
        :return: result.csv
        """
        test = pd.DataFrame(data=self.train_data, index=self.test_data)
        test = self.feature_engineering(test)

        test = pd.DataFrame(data=test, columns=self.feature_selection_list, index=self.test_data)

        test = self.feature_selection(test)

        d_matrix_test = xgb.DMatrix(test)
        y_predicted = self.model.predict(d_matrix_test)

        test_answer = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(y_predicted)], ignore_index=True, axis=1)
        final_result = pd.DataFrame(test_answer)
        final_result.to_csv('result/result.csv', index=False)

    def tpot_search(self):
        """
        Search model with TPOT
        :return: tpot_pipeline.py
        """
        train = pd.DataFrame(data=self.train_data, index=self.train_answer.keys()).values
        answer = [int(value) for value in self.train_answer.values()]

        x_train, x_test, y_train, y_test = train_test_split(
                                                    train.astype(np.float64),
                                                    answer,
                                                    train_size=0.75,
                                                    test_size=0.25
        )

        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        tpot.fit(x_train, y_train)
        tpot.export('tpot_model/tpot_pipeline.py')

