import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from tpot.builtins import StackingEstimator


class Model(object):
    def __init__(self):
        self.train_data = self.load('pickle/train_data_set.pkl')
        self.train_answer = self.load('pickle/train_answer.pkl')
        self.test_data = [test_id for test_id in pd.read_csv("data/test.csv", sep=',')['bidder_id']]
        self.model = None

    @staticmethod
    def load(pickle_name):
        with open(pickle_name, 'rb') as data_file:
            data = pickle.load(data_file)
        return data

    def return_statistical_data(self):
        return self.train_data[['nb_unique_ip', 'low_freq_ip', 'high_freq_ip', 'std_freq_ip',
                                'nb_unique_device', 'low_freq_device', 'high_freq_device', 'std_freq_device',
                                'nb_unique_url', 'low_freq_url', 'high_freq_url', 'std_freq_url',
                                'nb_unique_auction', 'low_freq_auction', 'high_freq_auction', 'std_freq_auction']]

    def train(self):

        train = pd.DataFrame(data=self.return_statistical_data(), index=self.train_answer.keys()).values
        answer = [int(value) for value in self.train_answer.values()]

        d_matrix_train = xgb.DMatrix(train, answer)

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

        num_boost_rounds = 1250

        self.model = xgb.train(dict(xgb_params, silent=0), d_matrix_train, num_boost_round=num_boost_rounds)

    def tpot_search(self):
        train = pd.DataFrame(data=self.return_statistical_data(), index=self.train_answer.keys()).values
        answer = [np.float64(value) for value in self.train_answer.values()]

        X_train, X_test, y_train, y_test = train_test_split(
                                                    train.astype(np.float64),
                                                    answer,
                                                    train_size=0.75,
                                                    test_size=0.25
        )

        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        tpot.fit(X_train, y_train)
        tpot.export('src/tpot_pipeline.py')
        # print(tpot.score(X_test, y_test))

    def tpot_train_test(self):
        train = pd.DataFrame(data=self.return_statistical_data(), index=self.train_answer.keys()).values
        answer = [np.float64(value) for value in self.train_answer.values()]
        test = pd.DataFrame(data=self.return_statistical_data(), index=self.test_data).values

        training_features, testing_features, training_target, testing_target = train_test_split(
            train,
            answer,
            random_state=42
        )

        imputer = Imputer(strategy="median")
        imputer.fit(training_features)
        training_features = imputer.transform(training_features)
        test = imputer.transform(test)

        # Score on the training set was:0.9549382851862445
        exported_pipeline = make_pipeline(
            StackingEstimator(estimator=ExtraTreesClassifier(
                bootstrap=False,
                criterion="gini",
                max_features=0.4,
                min_samples_leaf=12,
                min_samples_split=17,
                n_estimators=100)
            ),
            GradientBoostingClassifier(
                learning_rate=0.01,
                max_depth=4,
                max_features=0.4,
                min_samples_leaf=6,
                min_samples_split=7,
                n_estimators=100,
                subsample=0.45
            )
        )

        exported_pipeline.fit(training_features, training_target)
        results = exported_pipeline.predict(test)
        test_answer = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(results)], ignore_index=True, axis=1)
        final_result = pd.DataFrame(test_answer)
        print(final_result)
        final_result.to_csv('result/result.csv', index=False)

    def test(self):
        test = pd.DataFrame(data=self.return_statistical_data(), index=self.test_data).values
        dtest = xgb.DMatrix(test)
        y_pred = self.model.predict(dtest)
        test_answer = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(y_pred)], ignore_index=True, axis=1)
        final_result = pd.DataFrame(test_answer)
        print(final_result)
        final_result.to_csv('result/result.csv', index=False)