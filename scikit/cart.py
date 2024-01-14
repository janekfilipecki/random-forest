import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

#df_read = pd.read_csv('../datasets/heart.csv')
#df_read = pd.read_csv('../datasets/WineQuality.csv')

data_read = pd.read_csv('../datasets/Parkinsson disease.csv')
data = data_read.drop(columns=['name'])

print(data.head())

data_final = pd.get_dummies(data,drop_first=True)

print(data_final.head())


# split
X = data_final.drop('status',axis=1)
y = data_final['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# print(data_final.info())
# print(X_train.info())

# Decision Tree + stats
tree = DecisionTreeClassifier(criterion='gini',max_depth=None)
tree.fit(X_train,y_train)

predictions = tree.predict(X_test)
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)
print(cm)