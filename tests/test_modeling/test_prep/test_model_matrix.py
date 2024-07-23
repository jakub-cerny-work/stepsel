import numpy as np
import pandas as pd
import pytest
from stepsel.modeling.prep import prepare_model_matrix

@pytest.fixture
def data():
    # Sample data
    data = {
        "y": list(range(1, 11)),
        "var1": list(range(1, 11)),  # Numerical
        "var2": list(range(11, 21)),  # Numerical
        "cat1": pd.Categorical(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]),  # Categorical
        "cat2": pd.Categorical(["X", "Y", "Z", "X", "Y", "Y", "Z", "Z", "X", "Y"]),  # Categorical
        "x1": list(range(21, 31)),  # Numerical
        "x2": pd.Categorical(["C", "D", "C", "D", "C", "D", "C", "D", "C", "D"]),  # Categorical
        "break": ["U", "V", "U", "V", "U", "V", "U", "V", "U", "V"]
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("formula, expected_y, expected_model_matrix_first_row, expected_model_matrix_second_row, expected_model_matrix_columns, expected_feature_ids", [
    ("y ~ var1 + var2 + x1", list(range(1, 11)), [1, 1, 11, 21], [1, 2, 12, 22], ["Intercept", "var1", "var2", "x1"], ["Intercept", "var1", "var2", "x1"]),
    ("y ~ cat1 + cat2 + x2", list(range(1, 11)), [1, 0, 0, 0, 0], [1, 1, 1, 0, 1], ['Intercept', 'cat1: B', 'cat2: Y', 'cat2: Z', 'x2: D'], ['Intercept', 'cat1', 'cat2', 'cat2', 'x2']),
    ("y ~ var1 + cat1 + x2", list(range(1, 11)), [1, 1, 0, 0], [1, 2, 1, 1], ['Intercept', 'var1', 'cat1: B', 'x2: D'], ["Intercept", "var1", "cat1", "x2"]),
    ("y ~ var1 * var2", list(range(1, 11)), [1, 11], [1, 24], ["Intercept", "var1 * var2"], ["Intercept", "var1 * var2"]),
    ("y ~ cat1 * cat2", list(range(1, 11)), [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0], ['Intercept', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z'], ['Intercept', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2']),
    ("y ~ var1 * cat1 + var2 * x2", list(range(1, 11)), [ 1,  1,  0, 11,  0], [ 1,  0,  2,  0, 12], ['Intercept', 'cat1: A * var1', 'cat1: B * var1', 'x2: C * var2', 'x2: D * var2'], ['Intercept', 'var1 * cat1', 'var1 * cat1', 'var2 * x2', 'var2 * x2']),
    ("y ~ var1", list(range(1, 11)), [1, 1], [1, 2], ['Intercept', 'var1'], ['Intercept', 'var1']),
    ("y ~ cat1", list(range(1, 11)), [1, 0], [1, 1], ['Intercept', 'cat1: B'], ['Intercept', 'cat1']),
    ("y ~ var1 * var2 + var1 * cat1 + cat2 * var2 + cat1 * cat2 + x1 + x2", list(range(1, 11)), [ 1, 21,  0, 11,  1,  0, 11,  0,  0,  0,  0,  0,  0,  0], [ 1, 22,  1, 24,  0,  2,  0, 12,  0,  0,  0,  0,  1,  0], ['Intercept', 'x1', 'x2: D', 'var1 * var2', 'cat1: A * var1', 'cat1: B * var1', 'cat2: X * var2', 'cat2: Y * var2', 'cat2: Z * var2', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z'], ['Intercept', 'x1', 'x2', 'var1 * var2', 'var1 * cat1', 'var1 * cat1', 'cat2 * var2', 'cat2 * var2', 'cat2 * var2', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2', 'cat1 * cat2']),
])
def test_prepare_model_matrix_formulas(data, formula, expected_y, expected_model_matrix_first_row, expected_model_matrix_second_row, expected_model_matrix_columns, expected_feature_ids):
    y, model_matrix, feature_ids = prepare_model_matrix(formula, data)
    assert np.all(y.iloc[:,0] == expected_y), "Response variable does not match expected value"
    assert np.all(model_matrix.iloc[0].values == expected_model_matrix_first_row), "First row of model matrix does not match expected values"
    assert np.all(model_matrix.iloc[1].values == expected_model_matrix_second_row), "Second row of model matrix does not match expected values"
    assert np.all(model_matrix.columns == expected_model_matrix_columns), "Column names of model matrix do not match expected values"
    assert np.all(feature_ids == expected_feature_ids), "Feature IDs do not match expected values"

@pytest.mark.parametrize("intercept, drop_first, expected_columns", [
    (True, False, ['Intercept', 'x1', 'x2: C', 'x2: D', 'var1 * var2', 'cat1: A * var1', 'cat1: B * var1', 'cat2: X * var2', 'cat2: Y * var2', 'cat2: Z * var2', 'cat1 * cat2: A * X', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z']),
    (False, True, ['x1', 'x2: D', 'var1 * var2', 'cat1: A * var1', 'cat1: B * var1', 'cat2: X * var2', 'cat2: Y * var2', 'cat2: Z * var2', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z']),
    (False, False, ['x1', 'x2: C', 'x2: D', 'var1 * var2', 'cat1: A * var1', 'cat1: B * var1', 'cat2: X * var2', 'cat2: Y * var2', 'cat2: Z * var2', 'cat1 * cat2: A * X', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z']),
])
def test_prepare_model_matrix_arguments(data, intercept, drop_first, expected_columns):
    formula = "y ~ var1 * var2 + var1 * cat1 + cat2 * var2 + cat1 * cat2 + x1 + x2"
    y, model_matrix, feature_ids = prepare_model_matrix(formula, data, intercept=intercept, drop_first=drop_first)
    assert np.all(model_matrix.columns == expected_columns), "Column names of model matrix do not match expected values"

def test_prepare_model_matrix_omit_left(data):
    formula = "y ~ var1 * var2 + var1 * cat1 + cat2 * var2 + cat1 * cat2 + x1 + x2"
    expected_columns = ['Intercept', 'x1', 'x2: D', 'var1 * var2', 'cat1: A * var1', 'cat1: B * var1', 'cat2: X * var2', 'cat2: Y * var2', 'cat2: Z * var2', 'cat1 * cat2: A * Y', 'cat1 * cat2: A * Z', 'cat1 * cat2: B * X', 'cat1 * cat2: B * Y', 'cat1 * cat2: B * Z']
    model_matrix, feature_ids = prepare_model_matrix(formula, data, omit_left_side_variables=True)
    assert np.all(model_matrix.columns == expected_columns), "Column names of model matrix do not match expected values"

def test_prepare_model_matrix_invalid_interaction(data):
    formula = "y ~ var1 * break"
    with pytest.raises(ValueError):
        prepare_model_matrix(formula, data)







# Test adjust_model_matrix function
from stepsel.modeling.prep.model_matrix import adjust_model_matrix

@pytest.fixture
def setup_data():
    # Create sample data for testing
    df1 = pd.DataFrame({'cat1: A': [1, 0, 0], 'cat1: B': [0, 1, 0], 'cat1: C': [0, 0, 1]})
    df2 = pd.DataFrame({'cat1: A': [0, 1, 0], 'cat1: B': [1, 0, 0], 'cat1: C': [0, 0, 1]})
    offsets = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    adjusted_coeffs = {'cat1: A': 0.2, 'cat1: B': -0.5}
    expected_df1 = pd.DataFrame({'cat1: C': [0, 0, 1]})
    expected_df2 = pd.DataFrame({'cat1: C': [0, 0, 1]})
    expected_offsets = [np.array([0.2, -0.5, 0]), np.array([-0.5, 0.2, 0])]
    model_matrices = [df1, df2]
    return model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets

def test_valid_adjustment(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    adj_df1, adj_df2, adj_off1, adj_off2 = adjust_model_matrix(model_matrices, adjusted_coeffs, offsets)
    assert adj_df1.equals(expected_df1), "First matrix not adjusted correctly"
    assert adj_df2.equals(expected_df2), "Second matrix not adjusted correctly"
    assert np.all(np.array(adj_off1) == expected_offsets[0]), "First offset not adjusted correctly"
    assert np.all(np.array(adj_off2) == expected_offsets[1]), "Second offset not adjusted correctly"

def test_no_offsets_provided(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    adj_df1, adj_df2, adj_off1, adj_off2 = adjust_model_matrix(model_matrices, adjusted_coeffs)
    assert adj_df1.equals(expected_df1), "First matrix not adjusted correctly"
    assert adj_df2.equals(expected_df2), "Second matrix not adjusted correctly"
    assert np.all(np.array(adj_off1) == expected_offsets[0]), "First offset not adjusted correctly"
    assert np.all(np.array(adj_off2) == expected_offsets[1]), "Second offset not adjusted correctly"

def test_mismatched_offsets_and_matrices_length(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    with pytest.raises(Exception):
        adjust_model_matrix(model_matrices, adjusted_coeffs, offsets[:-1])

def test_mismatched_rows_in_matrix_and_offset(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    offsets = [np.array([0.5]), np.array([1.5, 2.0])]
    with pytest.raises(Exception):
        adjust_model_matrix(model_matrices, adjusted_coeffs, offsets)

def test_adjustment_with_empty_coefficients(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    adj_df1, adj_df2, adj_off1, adj_off2 = adjust_model_matrix(model_matrices, {}, offsets)
    assert adj_df1.equals(model_matrices[0]), "First matrix not adjusted correctly"
    assert adj_df2.equals(model_matrices[1]), "Second matrix not adjusted correctly"
    assert np.all(np.array(adj_off1) == offsets[0]), "First offset not adjusted correctly"
    assert np.all(np.array(adj_off2) == offsets[1]), "Second offset not adjusted correctly"

def test_adjustment_with_nonexistent_variables(setup_data):
    model_matrices, adjusted_coeffs, offsets, expected_df1, expected_df2, expected_offsets = setup_data
    adjusted_coeffs['C'] = 1.0  # Nonexistent variable
    with pytest.raises(KeyError):
        adjust_model_matrix(model_matrices, adjusted_coeffs, offsets)
