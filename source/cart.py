import pandas as pd
import typing
import copy
import numpy as np
import math
from operator import xor


class CART:
    def __init__(
        self,
        #  HYPERPARAMETERS
        max_depth: int = math.inf,
        min_rows: int = 2,
        split_search_density: int = 10,
        min_impurity_treshold: float = 0,
        selective_pruning: bool = True,
    ):
        """This class represents a CART decision tree. It is a control
        class that uses the Node class to build a decision tree and perform
        predictions.

        Hyperparameters:
            - max_depth: The maximum depth of the tree for pruing purposes. By
                default it is set to infinity, meaning that the tree will grow
                until all end conditions are met.
            - min_rows: The minimum amount of rows that a node must have in
            order to be able to split. 2 by default.
            - split_search_density: The density of the split search. The
                higher the density, the more splits will be calculated. 10 by
                default.
            - min_impurity_treshold=: The gini impurity treshold for pruing
                purposes. The
                impurity pruning is off by default.
            - selective_pruning: If selective pruning is on, the node will only
                be pruned if the gini impurity of the best split is not better
                than the parent gini impurity. True by default.

        Example:
            TODO
        Attributes:
            TODO

        """
        self._max_depth: int = max_depth
        self._min_rows: int = min_rows
        self._split_search_density: int = split_search_density
        self._min_impurity_treshold: float = min_impurity_treshold
        self._selective_pruning: bool = selective_pruning

        # The root node of the tree, set up when calling the fit method
        self._root: Node = None

    def __repr__(self) -> str:
        """This method represents the object as a string.

        Args:
            None
        Returns:
            (str) A string representation of the object
        Raises:
            None
        """
        return repr(self._root)

    def fit(
        self, data, target_feature, categorical_features: typing.List[str] = []
    ):
        """This method builds the decision tree using the Node class.

        Args:
            None
        Returns:
            None
        Raises:
            TODO
        """
        # Create the root node
        self._root = Node(
            data=data,
            target_feature=target_feature,
            categorical_features=categorical_features,
            max_depth=self._max_depth,
            min_rows=self._min_rows,
            split_search_density=self._split_search_density,
            min_impurity_treshold=self._min_impurity_treshold,
            selective_pruning=self._selective_pruning,
        )

        # Grow the root node
        self._root._grow()

    def predict(self, row) -> object:
        """This method predicts the class of a row.

        Args:
            row (pd.Series): The row that the prediction will be made on
        Returns:
            (object) The predicted value of the target feature
        Raises:
            TODO
        """
        return self._root._predict(row)


class Node:
    def __init__(
        self,
        data: pd.DataFrame,
        target_feature: str,
        categorical_features: typing.List[str],
        depth: int = 0,
        parent_split_impurity: float = None,
        #  HYPERPARAMETERS
        max_depth: int = math.inf,
        min_rows: int = 2,
        split_search_density: int = 10,
        min_impurity_treshold: float = 0,
        selective_pruning: bool = True,
    ):
        """This class represents a node in a decision tree. It provides
        an necessary logic to build a decision tree and perform predictions.

        Example:
            TODO
        Attributes:
            TODO

        """

        # The data must be in pandas format, without indexing columns.
        self._data: pd.DataFrame = data
        # The target feature is the one that we are trying to predict.
        self._target_feature: str = target_feature
        # Categorical features are those that cannot be treated with
        # comparison operators. For coherence this is supposed to be a
        # list of strings.
        self._categorical_features: typing.List(str) = categorical_features
        # The depth of the node. The root node has a depth of 0.
        self._depth: int = depth
        # In this implementation, we are only allowing binary splits
        # so if the feature is categorical, we will split the data into
        # two subsets, one for the chosen feature and the other for all
        # other values
        self._current_split_feature: str = None
        # Just to reduce the amount of if statements
        self._current_split_feature_is_categorical: bool = None
        self._current_split_value: str or float = None
        # In case of a categorical feature, the left child will be the one
        # that contains the chosen split value.
        self._left_child: Node = None
        self._right_child: Node = None
        # The gini impurity of the parent node. If the Node is the root node,
        # this value will be None.
        self._parent_split_impurity: float = parent_split_impurity

        # The predicted value of the node. If the node is not a leaf node,
        # this value will be None.
        self._leaf_label: object = None

        # HYPERPARAMETERS

        # The maximum depth of the tree for pruing purposes
        self._max_depth: int = max_depth
        # The minimum amount of rows that a node must have in order
        # to be able to split
        self._min_rows: int = min_rows
        # The density of the split search. The higher the density, the more
        # splits will be calculated
        self._split_search_density: int = split_search_density
        # The gini impurity treshold for pruing purposes
        self._min_impurity_treshold: float = min_impurity_treshold
        # If selective pruning is on, the node will only be pruned if the
        # gini impurity of the best split is not better than the parent
        # gini impurity
        self._selective_pruning: bool = selective_pruning

    def __repr__(self) -> str:
        """This method represents the object as a string.

        Args:
            None
        Returns:
            None
        Raises:
            None
        """

        tabs = "|    " * self._depth
        output = tabs
        if self._left_child and self._right_child:
            output += "Feature: " + self._current_split_feature + "\n"
            if self._current_split_feature_is_categorical:
                output += tabs + "== " + str(self._current_split_value) + "\n"
                output += repr(self._left_child)
                output += tabs + "!= " + str(self._current_split_value) + "\n"
                output += repr(self._right_child)
            else:
                output += tabs + "<= " + str(self._current_split_value) + "\n"
                output += repr(self._left_child)
                output += tabs + "> " + str(self._current_split_value) + "\n"
                output += repr(self._right_child)
        else:
            output += "Label: " + str(self._leaf_label) + "\n"

        return output

    def _grow_leaf(self) -> None:
        """This method sets the label of the leaf node to the most common value

        Args:
            None
        Returns:
            None
        Raises:
            TODO
        """
        # Set the label of the node to the most common value of the target
        # feature
        self._leaf_label = self._data[self._target_feature].mode()[0]

    def _gini(self, subset: pd.DataFrame) -> float:
        """Calculate the gini impurity of a subset of the nodes data.

        Args:
            subset (pd.DataFrame): The subset of the nodes data
        Returns:
            (int) The gini impurity of the subset
        Raises:
            TODO
        """

        # Initialize the gini impurity with one
        gini = 1
        # Get the size of the subset
        subset_size = subset.shape[0]
        # For each unique value in the feature, calculate the probability
        # of that value appearing in the subset
        for value in subset[self._target_feature].unique():
            # Calculate the amount of rows that have the value
            # in the target feature
            value_appearances = subset[
                subset[self._target_feature] == value
            ].shape[0]
            # Calculate the probability of the value appearing
            value_probability = value_appearances / subset_size
            # Subtract the probability squared from the gini impurity,
            # as per the formula
            gini -= value_probability**2

        return gini

    def _get_subsets(
        self, split_feature: str, is_categorical: bool, split_value: object
    ) -> typing.List[pd.DataFrame]:
        """Get the subsets of the nodes data based on the split feature and
        split value.

        Args:
            split_feature (str): The feature that will be used to split the
                data
            split_value (str or float): The value that will be used to split
                the data. If the feature is categoricalso we will split the
                data into two subsets, one for the chosen feature and the
                other for all other values. In case of a categorical feature,
                the left child will be the one that contains the chosen split
                value.

        Returns:
            (list) A list of subsets of the nodes data
        Raises:
            TODO
        """
        # If the feature is categorical
        if is_categorical:
            left_subset = self._data[self._data[split_feature] == split_value]
            right_subset = self._data[self._data[split_feature] != split_value]
            return [left_subset, right_subset]
        # If the feature is numerical
        else:
            # Left is inclusive, right is exclusive
            left_subset = self._data[self._data[split_feature] <= split_value]
            right_subset = self._data[self._data[split_feature] > split_value]
            return [left_subset, right_subset]

    def _get_splits(self) -> typing.List[typing.Tuple[str, bool, object]]:
        """Get the possible splits of the nodes data.
        Getting the possible splits is very expensive. A better implementation
        would be to calculate the gini impurity while iterating through the
        numerical features (it doesnt change anything for categorical
        features). This would require a different implementation of the _gini
        method and the code would be way less readable, thus we decided to
        keep it this way.

        Args:
            None
        Returns:
            (list(tuple(str, bool, str or float))) A list of tuples containing
                the split feature and split value
        Raises:
            TODO
        """
        splits = []
        # Remove the target feature from the list of features
        features = copy.copy(self._data.columns.to_list())
        features.remove(self._target_feature)
        # For each feature
        for feature in features:
            is_categorical = False
            # If the feature is categorical
            if feature in self._categorical_features:
                is_categorical = True
                # For each value in the feature
                for split_value in self._data[feature].unique():
                    splits.append((feature, is_categorical, split_value))
            # If the feature is numerical
            else:
                # Here is the tricky part. We can assume that splitting the
                # data in every possible value of the feature will give us
                # the best split, that is by every integer between the minimum
                #  and maximum. However, this is not true. The best split
                #  might be in a value that is not an integer. So we will
                #  split the data by creating an evenly spaced array of a
                # fixed size and then we will calculate the gini impurity
                # for each value in the array. This value is a hyperparameter.
                max_value = self._data[feature].max()
                min_value = self._data[feature].min()
                split_values = np.linspace(
                    min_value, max_value, self._split_search_density
                )
                for split_value in split_values:
                    splits.append((feature, is_categorical, split_value))

        return splits

    def _rate_split(self, subsets: typing.List[pd.DataFrame]) -> float:
        """This function represents calculating the gini impurity of a split,
        but we won't be subtracting the gini impurity of the subsets from the
        gini impurity of the parent node, since we are only interested in the
        gini impurity of the subsets. The lower the gini impurity of the
        subsets, the better the split. So this function will effectively
        only return a weighted average of the gini impurities of the subsets.

        Args:
            subsets (list(pd.DataFrame)): The subsets of the nodes data
        Returns:
            (int) The gini impurity of the subsets.
        Raises:

        """
        weighted_gini = 0

        for subset in subsets:
            # Calculate the gini impurity of the subset and multiply it by the
            # size of the subset divided by the size of the nodes data
            # effectively getting the weighted average of the gini impurities
            weighted_gini += (
                self._gini(subset) * subset.shape[0] / self._data.shape[0]
            )

        return weighted_gini

    def _grow(self) -> None:
        """This method provides the logic to grow the tree. In current
        implementation it creates two children nodes if None of the end
        conditions or pruning conditions are met.

        The end conditions are:
            - The node is pure
            - The node has no more features to split

        The pruning conditions are:
            - The node has less than a certain amount of rows
            - The max depth is reached
            - The gini impurity of the parent is less than a certain treshold
            - The gini impurity of the best split is not better than the parent

        Args:
            None
        Returns:
            None
        Raises:
            TODO
        """
        # Check end conditions

        # If the node is pure (or there are no rows left)
        if (
            self._gini(self._data) == 0
            or
            # If the node has no more features to split
            len(self._data.columns.to_list()) == 1
            or
            # Check pruning conditions
            # If the node has less than a certain amount of rows
            self._data.shape[0] < self._min_rows
            or
            # If the max depth is reached
            self._depth >= self._max_depth
            or
            # If the gini impurity of the parent is less than a certain
            # treshold, given selective pruning is on
            (
                self._selective_pruning
                and self._parent_split_impurity is not None
                and self._parent_split_impurity < self._min_impurity_treshold
            )
        ):
            # Grow a leaf
            self._grow_leaf()
            return

        # Get the possible splits
        splits = self._get_splits()
        # Initialize the best split with None
        best_split = None
        # Initialize the best split gini with one
        best_split_gini = 1
        # For each split
        for split in splits:
            # Get the subsets of the split
            subsets = self._get_subsets(*split)
            # Calculate the gini impurity of the split
            split_gini = self._rate_split(subsets)
            # If the gini impurity of the split is better than the best split
            if split_gini < best_split_gini:
                # Update the best split
                best_split = split
                # Update the best split gini
                best_split_gini = split_gini

        # Check the last pruning condition

        # If the best split gini index is not better than the parent
        if (
            self._parent_split_impurity is not None
            and best_split_gini >= self._parent_split_impurity
        ):
            # Grow a leaf
            self._grow_leaf()
            return

        # Setup the node for predictions
        split_feature, is_categorical, split_value = best_split
        self._current_split_feature = split_feature
        self._current_split_feature_is_categorical = is_categorical
        self._current_split_value = split_value

        # Get the subsets for the children nodes
        left_data, right_data = self._get_subsets(*best_split)
        # Drop the split feature from the data
        left_data = left_data.drop(columns=[split_feature])
        right_data = right_data.drop(columns=[split_feature])

        # Create the children nodes
        self._left_child = Node(
            data=left_data,
            target_feature=self._target_feature,
            categorical_features=self._categorical_features,
            depth=self._depth + 1,
            parent_split_impurity=best_split_gini,
            max_depth=self._max_depth,
            min_rows=self._min_rows,
            split_search_density=self._split_search_density,
            min_impurity_treshold=self._min_impurity_treshold,
            selective_pruning=self._selective_pruning,
        )
        self._right_child = Node(
            data=right_data,
            target_feature=self._target_feature,
            categorical_features=self._categorical_features,
            depth=self._depth + 1,
            parent_split_impurity=best_split_gini,
            max_depth=self._max_depth,
            min_rows=self._min_rows,
            split_search_density=self._split_search_density,
            min_impurity_treshold=self._min_impurity_treshold,
            selective_pruning=self._selective_pruning,
        )

        # Grow the children nodes
        self._left_child._grow()
        self._right_child._grow()

    def _predict(self, row: pd.Series) -> object:
        """This method provides the logic to predict a row's class. It will
        return either call upon children nodes or return the most common
        value of the target feature.

        Args:
            row (pd.Series): The row that the prediction will be made on
        Returns:
            (object) The predicted value of the target feature
        Raises:
            TODO
        """

        # This assertion is here to make sure that there is no situation
        # where the node has only one child. This should never happen.
        assert not xor(bool(self._left_child), bool(self._right_child))

        # If the node is a leaf node (has no children)
        if not (self._left_child or self._right_child):
            # Return the most common value of the target feature
            return self._leaf_label

        # If the feature is categorical
        if self._current_split_feature_is_categorical:
            # If the row has the split value in the split feature
            if row[self._current_split_feature] == self._current_split_value:
                # Call the left child
                return self._left_child._predict(row)
            # If the row does not have the split value in the split feature
            else:
                # Call the right child
                return self._right_child._predict(row)
        # If the feature is numerical
        else:
            # If the row has the split value in the split feature
            if row[self._current_split_feature] <= self._current_split_value:
                # Predict the left child
                return self._left_child._predict(row)
            # If the row does not have the split value in the split feature
            else:
                # Predict the right child
                return self._right_child._predict(row)
