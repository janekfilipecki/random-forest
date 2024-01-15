import numpy as np
import pandas as pd
from cart import CART

def bootstrap_data(data):
    print(len(data))
    # draw with replacement
    bootstrap_indices = list(np.random.choice(range(len(data)), len(data), replace = True))
    # print("bootstrap_indices : " + str(bootstrap_indices))
    # print(len(bootstrap_indices))
    print("UNIQUE : " + str(len(np.unique(bootstrap_indices))))
    
    # out of bag -> Not drawn for the tree, can be used for error prediction
    oob_indices = [i for i in range(len(data)) if i not in bootstrap_indices]
    # TODO -> instead of deep copy can i not drop it from the original dataFrame?
    data_copy = data.copy(deep=True)
    data_bootstrap = data_copy.drop(oob_indices)
    # data_bootstrap = data.iloc[bootstrap_indices].values
    print("data : " + str(data_copy))
    print("data_bootstrap :" + str(data_bootstrap))
    return data_bootstrap


nr_trees = 10
data = pd.read_csv('../datasets/Parkinsson_disease.csv')
data = data.drop(['name'], axis=1)
cart = CART(split_search_density=5)




for i in range (0, 1):
    cart.fit(bootstrap_data(data), 'status', [])
    data['predictions'] = data.apply(cart.predict, axis=1)
    print(data)


# data['predictions'] = data.apply(cart.predict, axis=1)
# accuracy_score(data['status'], data['predictions'])
# data.head()





# TODO
# In a loop, where n = 10?
# Bootstrap the data -> select 2/3, 1/3 could be used for error prediction
# Train a tree on the bootstrapped data, get a prediction
# Array of predictions -> Get the most voted class