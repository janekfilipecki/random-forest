import pytest
from pytest import approx
from source.cart import Node
import pandas as pd

data1 = pd.DataFrame({
    'feature1': [1, 1, 0, 0, 1, 0, 1, 0],
    'target': [1, 0, 1, 0, 1, 0, 1, 1]
})


def test_node_init():
    node = Node(data1, 'target', data1.columns.to_list())
    assert node._data.equals(data1)
    assert node._target_feature == 'target'
    assert node._categorical_features == data1.columns.to_list()


def test_gini_impurity_full_dataset():
    node = Node(data1, 'target', data1.columns.to_list())
    gini = node._gini(data1)
    assert gini == approx(30/64)


def test_gini_impurity_subset():
    subset_indices = [1, 3, 5]
    node = Node(data1, 'target', data1.columns.to_list())
    gini = node._gini(data1.iloc[subset_indices, :])
    assert gini == approx(0)


def test_gini_one_row():
    subset_indices = [1]
    node = Node(data1, 'target', data1.columns.to_list())
    gini = node._gini(data1.iloc[subset_indices, :])
    assert gini == approx(0)


data2 = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    'feature2': [1, 0, 1, 0, 1, 0, 1, 0],
    'target': [1, 0, 1, 0, 1, 0, 1, 0]
})
categorical_features2 = ['feature2']


def test_get_subsets_categorical():
    node = Node(data2, 'target', categorical_features2)
    subsets = node._get_subsets('feature2', True, 1)
    assert subsets[0].equals(data2.iloc[[0, 2, 4, 6], :])
    assert subsets[1].equals(data2.iloc[[1, 3, 5, 7], :])


def test_get_subsets_numerical():
    node = Node(data2, 'target', categorical_features2)
    subsets = node._get_subsets('feature1', False, 5)
    assert subsets[0].equals(data2.iloc[[0, 1, 2, 3, 4], :])
    assert subsets[1].equals(data2.iloc[[5, 6, 7], :])


data3 = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': ["a", "b", "a", "b", "a", "c", "c", "b", "a", "b"],
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
})


def test_get_splits():
    node = Node(data3, 'target', ['feature2'])
    splits = node._get_splits()
    assert splits == [
        ('feature1', False, 1.),
        ('feature1', False, 2.),
        ('feature1', False, 3.),
        ('feature1', False, 4.),
        ('feature1', False, 5.),
        ('feature1', False, 6.),
        ('feature1', False, 7.),
        ('feature1', False, 8.),
        ('feature1', False, 9.),
        ('feature1', False, 10.),
        ('feature2', True, 'a'),
        ('feature2', True, 'b'),
        ('feature2', True, 'c')
    ]


