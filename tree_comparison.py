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

startTime = datetime.now()
rf = RandomForest(forest_size=50, selective_pruning=True)
rf.fit(data, target_feature='output')
data['predictions'] = data.apply(rf.predict, axis=1)
print(accuracy_score(data['output'], data['predictions']))
print(recall_score(data['output'], data['predictions']))
print(precision_score(data['output'], data['predictions']))
print(f1_score(data['output'], data['predictions']))
print("Time for trees = :" + str(datetime.now() - startTime)) 

# print(confusion_matrix(data['output'], data['predictions']))

# HEATMAP 
# sns.heatmap(confusion_matrix(data['output'], data['predictions'])/np.sum(confusion_matrix(data['output'], data['predictions'])), annot=True, 
#             fmt='.2%', cmap='Blues')
# plt.show()




# sns.heatmap(multilabel_confusion_matrix(data['quality'], data['predictions']))