from source.cart import CART
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_csv('datasets/Parkinsson_disease.csv')
data.head()
data = data.drop(['name'], axis=1)
data['status'].value_counts()
startTime = datetime.now()


cart = CART(split_search_density=5)
cart.fit(data, 'status', [])
data['predictions'] = data.apply(cart.predict, axis=1)
accuracy_score(data['status'], data['predictions'])
print("Time for single tree:" + str(datetime.now() - startTime)) 