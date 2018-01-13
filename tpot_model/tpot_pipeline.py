"""
TPOT model found with : generations=5, population_size=50
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Score on the training set was:0.9549382851862445
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=12, min_samples_split=17, n_estimators=100)),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=4, max_features=0.4, min_samples_leaf=6, min_samples_split=7, n_estimators=100, subsample=0.45)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
