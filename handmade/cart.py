import numpy as np
from sklearn.datasets import load_iris


def entropy(data):
    class_counts = np.unique(data, return_counts=True)
    probs = class_counts[1] / len(data)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def information_gain(data, split_feature, split_value):
    left_data = data[data[:, split_feature] <= split_value]
    right_data = data[data[:, split_feature] > split_value]

    left_entropy = entropy(left_data)
    right_entropy = entropy(right_data)

    weighted_entropy = left_entropy * len(left_data) / len(data) + right_entropy * len(right_data) / len(data)

    information_gain = entropy(data) - weighted_entropy
    return information_gain

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
    print(left_data)
    print(right_data)

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
print(iris)
data = iris.data
labels = iris.target

# Create a decision tree
tree = create_tree(np.concatenate((data, labels.reshape([-1, 1])), axis=1))

print(np.concatenate((data, labels.reshape([-1, 1])), axis=1))


def print_tree(tree, level=0):
    if not tree:
        return

    print("level = "+str(level)+str(tree.value)+str(tree.feature))
    if (tree.left_child):
        print_tree(tree.left_child, level + 1)
    if (tree.right_child):
        print_tree(tree.right_child, level + 1)



# Print the decision tree
print(tree)

# print_tree(tree)