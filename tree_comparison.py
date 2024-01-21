from source.cart import CART
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from source.random_forest import RandomForest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from datetime import datetime



data = pd.read_csv('datasets/heart.csv')


# data = data.iloc[1: , :]
# data = data.drop(['name'], axis=1)
# print(data.head())
# print(data['output'].value_counts())

# Selective pruning!

# startTime = datetime.now()
# rf = RandomForest(forest_size=50, selective_pruning=True)
# rf.fit(data, target_feature='output')
# data['predictions'] = data.apply(rf.predict, axis=1)
# print(accuracy_score(data['output'], data['predictions']))
# print(recall_score(data['output'], data['predictions']))
# print(precision_score(data['output'], data['predictions']))
# print(f1_score(data['output'], data['predictions']))
# print("Time for trees = :" + str(datetime.now() - startTime)) 

# print(confusion_matrix(data['output'], data['predictions']))

# HEATMAP 
# sns.heatmap(confusion_matrix(data['output'], data['predictions'])/np.sum(confusion_matrix(data['output'], data['predictions'])), annot=True, 
#             fmt='.2%', cmap='Blues')
# plt.show()

# Max depth

def get_data_for_max_depth(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 6):
        print(value-2*i)
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, max_depth=value-2*i)
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

# print(get_data_for_max_depth(13))


def get_data_for_min_rows(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 5):
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, min_rows=value+2*i)
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

# 2 4 6 8 10
# print(get_data_for_min_rows(2))


def get_data_for_split_density(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 5):
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, split_search_density=value-3*i)
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

# print(get_data_for_split_density(15))

def get_data_for_impurity_threshhold(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 7):
        print(0.1*i)
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, min_impurity_treshold=value+(0.1*i))
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

# print(get_data_for_impurity_threshhold(0))


def get_data_for_bootstrapping(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 7):
        print(str(value-(0.1*i)))
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, max_samples=value-(0.1*i))
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

# print(get_data_for_bootstrapping(1))


def get_data_for_feature_bagging(value):
    acc = []
    recall = []
    precision = []
    f1 = []
    time=[]
    for i in range (0, 5):
        print(str(value-(0.1*i)))
        copy_data = data.copy(deep=True)
        startTime = datetime.now()
        rf = RandomForest(forest_size=50, max_features=value-(0.1*i))
        rf.fit(copy_data, target_feature='output')
        copy_data['predictions'] = copy_data.apply(rf.predict, axis=1)
        acc.append(accuracy_score(copy_data['output'], copy_data['predictions']))
        recall.append(recall_score(copy_data['output'], copy_data['predictions']))
        precision.append(precision_score(copy_data['output'], copy_data['predictions']))
        f1.append(f1_score(copy_data['output'], copy_data['predictions']))
        time.append((datetime.now() - startTime).total_seconds())
    return acc, recall, precision, f1, time

print(get_data_for_feature_bagging(1))


# sns.heatmap(multilabel_confusion_matrix(data['quality'], data['predictions']))