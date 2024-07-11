"""
Helper functions for binning. Smaller functions that are used in multiple binning functions.
"""
import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_tree_cut_points(clf: DecisionTreeRegressor | DecisionTreeClassifier, feature_names: ArrayLike | None = None):
    """Get the cut points of a decision tree.

    Parameters
    ----------
    clf : DecisionTreeRegressor or DecisionTreeClassifier
        The decision tree to get the cut points from.
    
    feature_names : array-like, optional
        The feature names of the decision tree. If None, the features are assumed to be integers.
    
    Returns
    -------
    feature_cut_points : dict
        A dictionary with the feature names as keys and the cut points as values.
    """
    features = clf.tree_.feature
    cut_points = clf.tree_.threshold

    if feature_names is not None:
        feature_cut_points = {feature_names[f]: np.sort(np.unique(cut_points[np.where(features == f)])) for f in features if f >= 0} # negative numbers are nodes without split
    else:
        feature_cut_points = {f: np.sort(np.unique(cut_points[np.where(features == f)])) for f in features if f >= 0} # negative numbers are nodes without split
        
    return feature_cut_points


def bin_values(data: ArrayLike, thresholds: ArrayLike, right=True) -> np.ndarray:
    """Bin data into bins based on thresholds.

    Parameters
    ----------
    data : array-like
        The input values to be binned.

    thresholds : array-like
        The thresholds to use for binning, ordered from smallest to largest.
    
    right : bool, optional
        Whether the intervals should be closed on the right (default) or left.
    
    Returns
    -------
    binned_values : array-like
        The binned values. String format is "(a, b]" if right=True, "[a, b)" if right=False.

    TODO
    ----
    * Add option to return pd.Categorical ordered by thresholds.
    """
    # Bin the data
    bins = np.digitize(data, thresholds, right=right)
    # Create string variable based on the bin intervals with brackets based on right parameter
    if right:
        bin_labels = [f"(-Inf, {thresholds[0]}]"] + [f'({thresholds[i-1]}, {thresholds[i]}]' for i in range(1, len(thresholds))] + [f"({thresholds[-1]}, Inf)"]
    else:
        bin_labels = [f"(-Inf, {thresholds[0]})"] + [f'[{thresholds[i-1]}, {thresholds[i]})' for i in range(1, len(thresholds))] + [f"[{thresholds[-1]}, Inf)"]
    result = np.array([bin_labels[i] for i in bins])
    return result
