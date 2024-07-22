import pandas as pd
import pytest
from stepsel.modeling.prep.interaction import (
    interaction_categorical_numerical,
    interaction_numerical_numerical,
    interaction_categorical_categorical
)

# Sample data
data = {
    "y": list(range(1, 11)),
    "var1": list(range(1, 11)),  # Numerical
    "var2": list(range(11, 21)),  # Numerical
    "cat1": pd.Categorical(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]),  # Categorical
    "cat2": pd.Categorical(["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"]),  # Categorical
    "x1": list(range(21, 31)),  # Numerical
    "x2": pd.Categorical(["C", "D", "C", "D", "C", "D", "C", "D", "C", "D"]),  # Categorical
    "break": ["U", "V", "U", "V", "U", "V", "U", "V", "U", "V"]
}
dt = pd.DataFrame(data)

# Test interaction_categorical_numerical
@pytest.mark.parametrize("categorical_series,numerical_series,expected_columns,expected_first_row_value,expectd_second_row_value", [
    (dt["cat1"], dt["var1"], ["cat1: A * var1", "cat1: B * var1"], [1, 0], [0, 2]),
    (dt["cat2"], dt["x1"], ["cat2: X * x1", "cat2: Y * x1"], [21, 0], [0, 22]),
    (dt["x1"], dt["cat2"], ["cat2: X * x1", "cat2: Y * x1"], [21, 0], [0, 22]),
])
def test_valid_interaction_categorical_numerical(categorical_series, numerical_series, expected_columns, expected_first_row_value, expectd_second_row_value):
    result_df = interaction_categorical_numerical(categorical_series, numerical_series)
    first_row_value = result_df.iloc[0].values
    second_row_value = result_df.iloc[1].values
    assert all([a == b for a, b in zip(result_df.columns, expected_columns)]), "Column names do not match expected interaction terms"
    assert all([a == b for a, b in zip(first_row_value, expected_first_row_value)]), "First row values do not match expected values"
    assert all([a == b for a, b in zip(second_row_value, expectd_second_row_value)]), "Second row values do not match expected values"

@pytest.mark.parametrize("series1,series2", [
    (dt["cat1"], dt["cat2"]),
    (dt["var1"], dt["var2"]),
])
def test_invalid_interaction_categorical_numerical(series1, series2):
    with pytest.raises(ValueError):
        interaction_categorical_numerical(series1, series2)

# Test interaction_categorical_categorical
@pytest.mark.parametrize("series1,series2,expected_series_name,expected_first_value,expected_second_value", [
    (dt["cat1"], dt["cat2"], "cat1 * cat2", "A * X", "B * Y"),
    (dt["cat2"], dt["cat1"], "cat2 * cat1", "X * A", "Y * B"),
])
def test_valid_interaction_categorical_categorical(series1, series2, expected_series_name, expected_first_value, expected_second_value):
    result_series = interaction_categorical_categorical(series1, series2)
    first_value = result_series.iloc[0]
    second_value = result_series.iloc[1]
    assert result_series.name == expected_series_name, "Series names do not match expected interaction term"
    assert first_value == expected_first_value, "First value does not match expected interaction term"
    assert second_value == expected_second_value, "Second value does not match expected interaction term"

@pytest.mark.parametrize("series1,series2", [
    (dt["var1"], dt["var2"]),
    (dt["cat1"], dt["var1"]),
])
def test_invalid_interaction_categorical_categorical(series1, series2):
    with pytest.raises(ValueError):
        interaction_categorical_categorical(series1, series2)

# Test interaction_numerical_numerical
@pytest.mark.parametrize("series1,series2,expected_series_name,expected_first_value,expected_second_value", [
    (dt["var1"], dt["var2"], "var1 * var2", 11, 24),
    (dt["var2"], dt["var1"], "var2 * var1", 11, 24),
    (dt["x1"], dt["x1"], "x1 * x1", 441, 484),
])
def test_valid_interaction_numerical_numerical(series1, series2, expected_series_name, expected_first_value, expected_second_value):
    result_series = interaction_numerical_numerical(series1, series2)
    first_value = result_series.iloc[0]
    second_value = result_series.iloc[1]
    assert result_series.name == expected_series_name, "Series names do not match expected interaction term"
    assert first_value == expected_first_value, "First value does not match expected interaction term"
    assert second_value == expected_second_value, "Second value does not match expected interaction term"

@pytest.mark.parametrize("series1,series2", [
    (dt["cat1"], dt["cat2"]),
    (dt["cat1"], dt["var1"]),
    (dt["var1"], dt["cat1"]),
])
def test_invalid_interaction_numerical_numerical(series1, series2):
    with pytest.raises(ValueError):
        interaction_numerical_numerical(series1, series2)

@pytest.mark.parametrize("series1,series2", [
    (dt["break"], dt["var1"]),
    (dt["cat1"], dt["break"]),
])
def test_invalid_series_type(series1, series2):
    with pytest.raises(ValueError):
        interaction_categorical_numerical(series1, series2)
        interaction_categorical_categorical(series1, series2)
        interaction_numerical_numerical(series1, series2)

@pytest.mark.parametrize("series1,series2", [
    (pd.Series([], dtype="category"), pd.Series([], dtype="float")),
    (pd.Series([], dtype="float"), pd.Series([], dtype="category")),
])
def test_empty_series(series1, series2):
    result_df_int_cat_num = interaction_categorical_numerical(series1, series2)
    result_df_int_cat_cat = interaction_categorical_numerical(series1, series2)
    result_df_int_num_num = interaction_categorical_numerical(series1, series2)
    assert result_df_int_cat_num.empty, "Result should be an empty DataFrame for empty input series for categorical and numerical series"
    assert result_df_int_cat_cat.empty, "Result should be an empty DataFrame for empty input series for categorical and categorical series"
    assert result_df_int_num_num.empty, "Result should be an empty DataFrame for empty input series for numerical and numerical series"
