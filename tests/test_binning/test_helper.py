import numpy as np
import pytest
from stepsel.binning.helper import bin_values, get_tree_cut_points
from stepsel.datasets import load_soccer_data

# Helper function to compare dictionaries
def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.array_equal(dict1[key], dict2[key]):
            return False
    return True

# Load the data
@pytest.fixture
def data():
    return load_soccer_data()

# Test bin_values function
def test_bin_values():
    data = np.array([0, 5, 10, 15])
    thresholds = np.array([5, 10])
    
    # Test with right=True
    expected_right = np.array(['(-Inf, 5]', '(-Inf, 5]', '(5, 10]', '(10, Inf)'])
    result_right = bin_values(data, thresholds, right=True)
    np.testing.assert_array_equal(result_right, expected_right, "Right=True case failed")
    
    # Test with right=False
    expected_left = np.array(['(-Inf, 5)', '[5, 10)', '[10, Inf)', '[10, Inf)'])
    result_left = bin_values(data, thresholds, right=False)
    np.testing.assert_array_equal(result_left, expected_left, "Right=False case failed")
    
    # Test with thresholds having single value & right=True
    single_threshold = np.array([10])
    expected_single = np.array(['(-Inf, 10]', '(-Inf, 10]', '(-Inf, 10]', '(10, Inf)'])
    result_single = bin_values(data, single_threshold, right=True)
    np.testing.assert_array_equal(result_single, expected_single, "Single threshold case failed")

    # Test with thresholds having single value & right=False
    single_threshold = np.array([10])
    expected_single = np.array(['(-Inf, 10)', '(-Inf, 10)', '[10, Inf)', '[10, Inf)'])
    result_single = bin_values(data, single_threshold, right=False)
    np.testing.assert_array_equal(result_single, expected_single, "Single threshold case failed")

    # Test with empty data
    data_empty = np.array([])
    expected_empty = np.array([])
    result_empty = bin_values(data_empty, thresholds, right=True)
    np.testing.assert_array_equal(result_empty, expected_empty, "Empty data case failed")
    
    # Test with empty thresholds
    thresholds_empty = np.array([])
    # Expecting ValueError when thresholds are empty
    with pytest.raises(ValueError, match="Thresholds should not be empty."):
        bin_values(data, thresholds_empty)


# Test get_tree_cut_points function
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import pytest
from stepsel.binning.helper import get_tree_cut_points

@pytest.mark.parametrize("feature_names, expected", [
    (["dangerous_attacks"], {'dangerous_attacks': [ 34.5,  38.5,  45.5,  80.5,  84.5,  90.5,  96.5, 104.5]}),
    (["attacks", "dangerous_attacks", "yellow", "yellow_opp", "red", "red_opp"], {'red_opp': [0.5], 'yellow': [2.5], 'attacks': [106., 141.], 'dangerous_attacks': [49.5, 68.5], 'yellow_opp': [3.5]})
])
def test_get_tree_cut_points_with_feature_names(feature_names, expected):
    # Create a simple decision tree
    data = load_soccer_data()
    X = data.loc[:, feature_names]
    y = data.loc[:, "goals"]
    clf = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
    clf.fit(X, y)
    
    # Test with feature names
    actual = get_tree_cut_points(clf, feature_names)
    assert compare_dictionaries(actual, expected), "Cut points do not match expected values."

@pytest.mark.parametrize("feature_names, expected", [
    (["dangerous_attacks"], {0: [ 34.5,  38.5,  45.5,  80.5,  84.5,  90.5,  96.5, 104.5]}),
    (["attacks", "dangerous_attacks", "yellow", "yellow_opp", "red", "red_opp"], {5: [0.5], 2: [2.5], 0: [106., 141.], 1: [49.5, 68.5], 3: [3.5]})
])
def test_get_tree_cut_points_without_feature_names(feature_names, expected):
    # Create a simple decision tree
    data = load_soccer_data()
    X = data.loc[:, feature_names]
    y = data.loc[:, "goals"]
    clf = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
    clf.fit(X, y)
    
    # Test without feature names
    actual = get_tree_cut_points(clf)
    assert compare_dictionaries(actual, expected), "Cut points do not match expected values."

def test_get_tree_cut_points_invalid_clf():
    # Pass a non-decision tree classifier to the function
    clf = None
    with pytest.raises(ValueError, match="clf should be a DecisionTreeRegressor or DecisionTreeClassifier."):
        get_tree_cut_points(clf)

def test_get_tree_cut_points_invalid_feature_names():
    # Mock a decision tree classifier and pass invalid feature_names
    clf = DecisionTreeClassifier()
    feature_names = "not array-like"  # Invalid type for feature_names
    with pytest.raises(ValueError, match="feature_names should be None or array-like."):
        get_tree_cut_points(clf, feature_names)
