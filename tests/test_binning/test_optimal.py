from stepsel.datasets import load_soccer_data
from sklearn import tree
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
from stepsel.binning.optimal import OptimalBinningUsingDecisionTreeRegressor
import pytest

# Helper function to compare dictionaries
def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.array_equal(dict1[key], dict2[key]):
            return False
    return True

@pytest.fixture
def data():
    return load_soccer_data()

def test_optimal_binning_no_split_01(data):
    feature_names = ["attacks"]
    X = data.loc[:, feature_names]
    y = data.loc[:, "goals"]

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='poisson',scoring="neg_mean_poisson_deviance", refit="neg_mean_poisson_defiance", max_depth=3)
    clf.fit(X, y)

    actual_predition = np.round(np.unique(clf.predict(X)), 6)
    expected_prediction = np.array([1.496139])
    assert np.array_equal(actual_predition, expected_prediction), "Prediction values do not match."
    
    actual_best_params = clf.Log.fit_log["cv_fit"].best_params_
    expected_best_params = {'ccp_alpha': 0.009325046115273516}
    assert compare_dictionaries(actual_best_params, expected_best_params), "Best parameters do not match."
    
    actual_cut_points = clf.cut_points
    expected_cut_points = {}
    assert compare_dictionaries(actual_cut_points, expected_cut_points), "Cut points do not match."

    with pytest.raises(ValueError, match="The model has not been fitted yet or the cut points are not available, e.g. no optimal cut points found.") as e:
        clf.bin_values(data["attacks"])


def test_optimal_binning_one_split_02(data):
    feature_names = ["dangerous_attacks"]
    X = data.loc[:, feature_names]
    y = data.loc[:, "ref_interv_opp"]

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='squared_error',scoring="neg_mean_squared_error", refit="neg_mean_squared_error", max_depth=3)
    clf.fit(X, y)

    actual_predition = np.round(np.unique(clf.predict(X)), 6)
    expected_prediction = np.array([17.862745, 20.59375])
    assert np.array_equal(actual_predition, expected_prediction), "Prediction values do not match."
    
    actual_best_params = clf.Log.fit_log["cv_fit"].best_params_
    expected_best_params = {'ccp_alpha': 0.7420023923912353}
    assert compare_dictionaries(actual_best_params, expected_best_params), "Best parameters do not match."
    
    actual_cut_points = clf.cut_points
    expected_cut_points = {'dangerous_attacks': [51.5]}
    assert compare_dictionaries(actual_cut_points, expected_cut_points), "Cut points do not match."

    binned_values = Series(clf.bin_values(data["dangerous_attacks"]))
    actual_bin_counts = np.array(binned_values.value_counts())
    expected_bin_counts = np.array([416,102])
    assert np.array_equal(actual_bin_counts, expected_bin_counts), "Bin counts do not match."

    actual_bin_labels = binned_values.value_counts().index.values
    expected_bin_labels = np.array(['(51.5, Inf)', '(-Inf, 51.5]'])
    assert np.array_equal(actual_bin_labels, expected_bin_labels), "Bin labels do not match."


def test_optimal_binning_one_split_with_kmeans_03(data):
    feature_names = ["dangerous_attacks"]
    X = data.loc[:, feature_names]
    y = data.loc[:, "ref_interv_opp"]

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='squared_error',scoring="neg_mean_squared_error", refit="neg_mean_squared_error", max_depth=3, n_clusters=10, max_grid_length=10)
    clf.fit(X, y)

    actual_predition = np.round(np.unique(clf.predict(X)), 6)
    expected_prediction = np.array([17.862745, 20.59375])
    assert np.array_equal(actual_predition, expected_prediction), "Prediction values do not match."
    
    actual_best_params = clf.Log.fit_log["cv_fit"].best_params_
    expected_best_params = {'ccp_alpha': 0.7420023923912353}
    assert compare_dictionaries(actual_best_params, expected_best_params), "Best parameters do not match."
    
    actual_cut_points = clf.cut_points
    expected_cut_points = {'dangerous_attacks': [51.5]}
    assert compare_dictionaries(actual_cut_points, expected_cut_points), "Cut points do not match."

    binned_values = Series(clf.bin_values(data["dangerous_attacks"]))
    actual_bin_counts = np.array(binned_values.value_counts())
    expected_bin_counts = np.array([416,102])
    assert np.array_equal(actual_bin_counts, expected_bin_counts), "Bin counts do not match."

    actual_bin_labels = binned_values.value_counts().index.values
    expected_bin_labels = np.array(['(51.5, Inf)', '(-Inf, 51.5]'])
    assert np.array_equal(actual_bin_labels, expected_bin_labels), "Bin labels do not match."


def test_optimal_binning_more_splits_04(data):
    feature_names = ["attacks"]
    X = np.array(data.loc[:, feature_names])
    y = np.array(data.loc[:, "dangerous_attacks"])

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='squared_error',scoring="neg_mean_squared_error", refit="neg_mean_squared_error", max_depth=3)
    clf.fit(X, y)
    clf.set_feature_names(feature_names)

    actual_predition = np.round(np.unique(clf.predict(X)), 6)
    expected_prediction = np.array([53.796791, 65.973214, 77.871508, 97.55])
    assert np.array_equal(actual_predition, expected_prediction), "Prediction values do not match."
    
    actual_best_params = clf.Log.fit_log["cv_fit"].best_params_
    expected_best_params = {'ccp_alpha': 6.720540761916695}
    assert compare_dictionaries(actual_best_params, expected_best_params), "Best parameters do not match."
    
    actual_cut_points = clf.cut_points
    expected_cut_points = {0: [102.5, 115.5, 144.5]}
    assert compare_dictionaries(actual_cut_points, expected_cut_points), "Cut points do not match."

    binned_values = Series(clf.bin_values(data["attacks"]))
    actual_bin_counts = np.array(binned_values.value_counts())
    expected_bin_counts = np.array([187,  179,   112, 40])
    assert np.array_equal(actual_bin_counts, expected_bin_counts), "Bin counts do not match."

    actual_bin_labels = binned_values.value_counts().index.values
    expected_bin_labels = np.array(['(-Inf, 102.5]', '(115.5, 144.5]', '(102.5, 115.5]', '(144.5, Inf)'])
    assert np.array_equal(actual_bin_labels, expected_bin_labels), "Bin labels do not match."


def test_optimal_binning_set_invalid_feature_names_05(data):
    feature_names = ["attacks"]
    X = np.array(data.loc[:, feature_names])
    y = np.array(data.loc[:, "dangerous_attacks"])

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='squared_error',scoring="neg_mean_squared_error", refit="neg_mean_squared_error", max_depth=3)
    clf.fit(X, y)

    with pytest.raises(TypeError, match="feature_names must be a list"):
        clf.set_feature_names("attacks")

"""
def test_optimal_binning_plot_tree_06(data):
    feature_names = ["attacks"]
    X = data.loc[:, feature_names]
    y = data.loc[:, "dangerous_attacks"]

    clf = OptimalBinningUsingDecisionTreeRegressor(criterion='squared_error',scoring="neg_mean_squared_error", refit="neg_mean_squared_error", max_depth=3)
    clf.fit(X, y)

    # Test that no error is raised when plot_tree is called
    clf.plot_tree((15,10))
"""
