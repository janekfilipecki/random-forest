import numpy as np
import pandas as pd
from source.cart import CART
from sklearn.metrics import accuracy_score, mean_squared_error
import math
import typing
from collections import Counter


class RandomForest:
    def __init__(
        self,
        # FOREST HYPERPARAMETERS
        forest_size: int = 100,
        enable_sample_bootstrapping: bool = True,
        max_samples: float = 1,
        enable_feature_bagging: bool = True,
        max_features: float = 1,
        regression_mode: bool = False,
        # TREE HYPERPARAMETERS
        max_depth: int = math.inf,
        min_rows: int = 2,
        split_search_density: int = 10,
        min_impurity_treshold: float = 0,
        selective_pruning: bool = True,
    ):
        """This class represents a random forest classifier. It is a
        ensemble learning algorithm, allowing a significant improvement
        in performance over classic decision tree approaches.

        Forest hyperparameters:
            - forest_size: This is the key prameter of the algorithm. It
                specifies how many weak learners will be built and trained
                before collecting scores and computing the overall output.
            - enable_sample_bootstrapping: This parameter controls whether
                the trees will be trained on a bootstrap sample of the data
                or not. This usually enables the model to have less overall
                variance.
            - max_samples: The maximum part of samples that will be used
                for training. 1, meaning whole dataset by default.
            - enable_feature_bagging: This parameter controls whether
                the trees will be constructed using all the features or not.
                This usually enables the model to have less overall variance.
            - max_features: The maximum part of features that will be used
                for training. 1, meaning all features by default.
            - regression_mode: Enable this if you want the tree outputs to be
                averaged instead of taking the mode. Suitable for regression
                tasks.

        Tree hyperparameters:
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
        self._forest_size: int = forest_size
        self._enable_sample_bootstrapping: bool = enable_sample_bootstrapping
        self._max_samples: float = max_samples
        self._enable_feature_bagging: bool = enable_feature_bagging
        self._max_features: float = max_features
        self._regression_mode: bool = regression_mode
        self._max_depth: int = max_depth
        self._min_rows: int = min_rows
        self._split_search_density: int = split_search_density
        self._min_impurity_treshold: float = min_impurity_treshold
        self._selective_pruning: bool = selective_pruning

        self._trees: typing.List[CART] = []
        self._oob_data: typing.List[pd.DataFrame] = []
        self._target_feature: str = None

    def fit(
        self, data, target_feature, categorical_features: typing.List[str] = []
    ):
        """This method builds the random forest classifier.

        Args:
            None
        Returns:
            None
        Raises:
            TODO
        """
        # Set target feature for scoring purposes
        self._target_feature = target_feature
        # Create trees
        for _ in range(self._forest_size):
            # Bootstrap
            tree_train_data = data
            if self._enable_sample_bootstrapping:
                tree_train_data = data.sample(
                    frac=self._max_samples, replace=True
                )
                # Add to oob list
                self._oob_data.append(data.drop(tree_train_data.index))
            # Feature bagging
            if self._enable_feature_bagging:
                tree_train_data = tree_train_data.sample(
                    frac=self._max_features, axis=1, replace=False
                )
            # Create tree
            tree = CART(
                max_depth=self._max_depth,
                min_rows=self._min_rows,
                split_search_density=self._split_search_density,
                min_impurity_treshold=self._min_impurity_treshold,
                selective_pruning=self._selective_pruning,
            )
            # Fit tree
            tree.fit(tree_train_data, target_feature, categorical_features)
            # Add to forest
            self._trees.append(tree)

    def predict(self, row):
        """This method predicts the class of a row.

        Args:
            row (pd.Series): The row that the prediction will be made on
        Returns:
            (object) The predicted value of the target feature
        Raises:
            TODO
        """
        predictions = []
        # Perform prediction on each tree
        for tree in self._trees:
            predictions.append(tree.predict(row))
        # If regression mode, return the mean of the predictions
        if self._regression_mode:
            return np.mean(predictions)
        # Else, return the mode of the predictions
        else:
            counter = Counter(predictions)
            return counter.most_common(1)[0][0]

    def score(self):
        """This method calculates the accuracy of the model
        on the oob samples of the trainging data.

        Args:
            None
        Returns:
            (float) The accuracy of the model for classification tasks
                and the mean squared error for regression tasks.
        Raises:
            TODO
        """

        # Create predictions DataFrame
        predictions_df = pd.DataFrame()
        prediction_columns = []
        for i, (tree_oob_data, tree) in enumerate(
            zip(self._oob_data.copy(), self._trees)
        ):
            # For each tree, perform predictions
            # and add them to the predictions DataFrame
            column_name = f"predictions_{i}"
            prediction_columns.append(column_name)
            tree_oob_data[column_name] = tree_oob_data.apply(
                tree.predict, axis=1
            )
            predictions_df = pd.merge(
                predictions_df,
                tree_oob_data[column_name],
                how="right",
                left_index=True,
                right_index=True,
            )

            predictions_df.loc[
                tree_oob_data.index, self._target_feature
            ] = tree_oob_data[self._target_feature].values

        if self._regression_mode:
            predictions_df["average_vote"] = predictions_df[
                prediction_columns
            ].mean(axis=1, skipna=True)
            mse = mean_squared_error(
                predictions_df[self._target_feature],
                predictions_df["average_vote"],
            )
            return mse
        else:
            # If regression is disabled, compute the majority vote
            predictions_df["majority_vote"] = predictions_df[
                prediction_columns
            ].mode(axis=1, dropna=True)[0]

            accuracy = accuracy_score(
                predictions_df[self._target_feature],
                predictions_df["majority_vote"],
            )
            return accuracy
