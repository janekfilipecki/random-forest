import numpy as np
import pandas as pd
from cart import CART
from sklearn.metrics import accuracy_score

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
    # print("data : " + str(data_copy))
    # print("data_bootstrap :" + str(data_bootstrap))
    return data_bootstrap


tree_limit = 10
data = pd.read_csv('../datasets/Parkinsson_disease.csv')
data = data.drop(['name'], axis=1)
cart = CART(split_search_density=5)


results = pd.DataFrame({})

for i in range (0, tree_limit):
    cart.fit(bootstrap_data(data), 'status', [])
    results[i] = data.apply(cart.predict, axis=1)


# tutaj average i konkatenacja do 1 kolumny
results['mean']=results.mean(axis=1)
results['round_up']=results.mean(axis=1).round()
print(results)
print("ACC SCORE after voting: ")
print(accuracy_score(data['status'], results['round_up']))