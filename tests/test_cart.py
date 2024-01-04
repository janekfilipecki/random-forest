import pytest
from source.cart import SplitSubsetDescriptor, Node
import pandas as pd
import operator


def test_node_default():
    node = Node()
    assert node._data == None


test_df = pd.DataFrame({'a': ['a', 'b', 'c', 'd', 'e'],
                        'b': [1, 2, 3, 4, 5]})


def test_split_categorical_in_operator_equal():
    split = SplitSubsetDescriptor('a', 'b')
    assert test_df.iloc[1] in split


def test_split_numerical_in_operator_greater_equal():
    split = SplitSubsetDescriptor('b', 3, operator.ge)
    assert test_df.iloc[1] not in split


def test_split_numerical_in_operator_less():
    split = SplitSubsetDescriptor('b', 4, operator.lt)
    assert test_df.iloc[1] in split


def test_split_categorical_in_list():
    split = SplitSubsetDescriptor('b', [1, 2, 3])
    assert test_df.iloc[1] in split


def test_split_categorical_in_out_of_list():
    split = SplitSubsetDescriptor('b', [3, 4])
    assert test_df.iloc[1] not in split


# def test_gini_index():
#     node = Node()
#     subset
#     target_feature_name


# def test_gini_index():
#     node = Node()
#     target_feature_name


# def test_node_split_categorical():

#     if row in Split()
#         subsets[Split()].add(row)


# def test_node_split_no_features():
#     pass

# def test_node_split_one_two_features():
#     pass
