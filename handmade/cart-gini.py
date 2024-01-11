import numpy as np
from sklearn.datasets import load_iris

def print_tree(tree, level=0):
    if not tree:
        return

    for key in tree:
        print(f"{level} | {key}: {tree[key]}")
        if tree[key]:
            print_tree(tree[key], level + 1)

def calculate_gini(data):
    class_counts = np.unique(data[:, -1], return_counts=True)
    probs = class_counts[1] / len(data)
    impurity = 1 - np.sum(probs**2)
    return impurity

def information_gain(data, split_feature, split_value):
    left_data = data[data[:, split_feature] <= split_value]
    right_data = data[data[:, split_feature] > split_value]

    left_gini = calculate_gini(left_data)
    right_gini = calculate_gini(right_data)

    parent_gini = calculate_gini(data)

    weighted_gini = left_gini * len(left_data) / len(data) + right_gini * len(right_data) / len(data)

    gini_gain = parent_gini - weighted_gini
    return gini_gain


def best_split(data):
    max_info_gain = 0
    best_feature = None
    best_value = None

    for feature in range(len(data[0])):
        for value in set(data[:, feature]):
            info_gain = information_gain(data, feature, value)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
                best_value = value

    return best_feature, best_value

def create_tree(data):
    labels = data[:, -1]

    if np.unique(labels).size == 1:
        return labels[0]

    feature, value = best_split(data)
    left_data = data[data[:, feature] <= value]
    right_data = data[data[:, feature] > value]

    left_child = create_tree(left_data)
    right_child = create_tree(right_data)

    return {
        'feature': feature,
        'value': value,
        'left_child': left_child,
        'right_child': right_child
    }

# Load the Iris dataset

iris = load_iris()
# print(iris)
data = iris.data
labels = iris.target

# Create a decision tree
tree = create_tree(np.concatenate((data, labels.reshape([-1, 1])), axis=1))

# Print the decision tree
print(tree)

print_tree(tree)