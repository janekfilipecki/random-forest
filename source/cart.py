import pandas as pd
import typing
import operator


class SplitSubsetDescriptor():
    """This is a class representing a split of a Node's data.
    It is used to determine if a row of data is in the split. The
    can be based on a numerical or categorical feature. Both types can
    be single values or lists of values. 

    Attributes:
        TODO
    """

    def __init__(self,
                 split_feature_name: str,
                 split_value: typing.Any,
                 comp_operator: operator = operator.eq):
        self._split_feature_name = split_feature_name
        self._split_value = split_value
        self._opreator = comp_operator

    def __contains__(self, row):
        """When used with a 'in' statement, this method will 
        return True if the Pandas row is in the split

        Args:
            row: the Pandas row to check
        Returns:
            True if the item is in the split, False otherwise
        Raises:
            TODO

        """
        feature_value = row[self._split_feature_name]
        if isinstance(self._split_value, typing.Iterable):
            if feature_value in self._split_value:
                return True
            return False
        else:
            if self._opreator(feature_value, self._split_value):
                return True
            return False


class Split():
    """Split represents a collection of split descriptors, 
    effectively allowing for a logical comparison of multiple
    splits and calcultaions on them.

    Attributes:
        TODO
    """

    def __init__(self, subset_descriptors) -> None:
        pass


class Node:
    def __init__(self,
                 data: pd.DataFrame = None,
                 features_left: typing.List[str] = None):
        self._data = data
        self._split_children_dict: typing.Dict[SplitSubsetDescriptor,
                                               'Node'] = None,
        self._features_left = features_left
        self._split_feature_name: str = None

    def _gini(self, y):
        """return the gini impurity of a subset of the nodes data,
        where the target feature is specified by target_feature_name

        Args:

        Returns:

        Raises:

        """
        pass

    def _get_splits(self) -> typing.List[SplitSubsetDescriptor]:
        """TODO

        Args:

        Returns:

        Raises:

        """
        pass

    def _get_subsets(self) -> typing.Dict[SplitSubsetDescriptor, pd.DataFrame]:
        """TODO

        Args:

        Returns:

        Raises:

        """
        pass

    def _split_tree(self):
        """TODO

        Args:

        Returns:

        Raises:

        """
        pass
