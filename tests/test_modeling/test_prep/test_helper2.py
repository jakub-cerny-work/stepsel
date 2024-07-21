import pytest
import numpy as np
from stepsel.modeling.prep.helper import get_interaction_type

# Helper function to compare dictionaries
def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.array_equal(np.sort(dict1[key]), np.sort(dict2[key])):
            return False
    return True

# Test get_interaction_type function
@pytest.mark.parametrize("interaction, numerical_vars, categorical_vars, expected", [
    # Test cases for numerical_numerical interactions
    ("a * b", ["a", "b"], [], "numerical_numerical"),
    ("x1 * x2", ["x1", "x2"], [], "numerical_numerical"),

    # Test cases for categorical_categorical interactions
    ("cat1 * cat2", [], ["cat1", "cat2"], "categorical_categorical"),
    ("gender * species", [], ["gender", "species"], "categorical_categorical"),

    # Test cases for numerical_categorical interactions
    ("a * cat", ["a"], ["cat"], "numerical_categorical"),
    ("age * gender", ["age"], ["gender"], "numerical_categorical"),

    # Test cases for categorical_numerical interactions
    ("cat * b", ["b"], ["cat"], "categorical_numerical"),
    ("species * weight", ["weight"], ["species"], "categorical_numerical"),
])
def test_get_interaction_type_valid(interaction, numerical_vars, categorical_vars, expected):
    assert get_interaction_type(interaction, numerical_vars, categorical_vars) == expected

@pytest.mark.parametrize("interaction, numerical_vars, categorical_vars, error_message", [
    # Test cases for interactions with no '*' character
    ("a b", ["a", "b"], [], "Interaction must contain one (and only one) '*' character."),
    ("x1x2", ["x1", "x2"], [], "Interaction must contain one (and only one) '*' character."),

    # Test cases for interactions with more than one '*' character
    ("a * b * c", ["a", "b", "c"], [], "Interaction must contain one (and only one) '*' character."),
    ("x1 * x2 * x3", ["x1", "x2", "x3"], [], "Interaction must contain one (and only one) '*' character."),

    # Test cases for interactions with variables not in lists
    ("a * c", ["a"], ["b"], "Interaction variables must be either numerical or categorical."),
    ("cat1 * cat3", [], ["cat1", "cat2"], "Interaction variables must be either numerical or categorical."),
])
def test_get_interaction_type_invalid(interaction, numerical_vars, categorical_vars, error_message):
    with pytest.raises(ValueError) as excinfo:
        get_interaction_type(interaction, numerical_vars, categorical_vars)
    assert error_message in str(excinfo.value)

# Test parse_model_formula function
from stepsel.modeling.prep.helper import parse_model_formula
@pytest.mark.parametrize("formula, expected", [
    # Test cases for valid formulas
    ("y ~ a + b + a * b", (["y"], ["a * b"], ["a", "b"])),
    ("response ~ predictor", (["response"], [], ["predictor"])),
    ("y ~ x1 + x2 * x3", (["y"], ["x2 * x3"], ["x1"])),
    ("outcome ~ factor1 * factor2 + covariate", (["outcome"], ["factor1 * factor2"], ["covariate"])),
    # Test cases with spaces and multiple interactions
    (" y ~ a + b * c + d ", (["y"], ["b * c"], ["a", "d"])),
    ("var1 ~ var2 * var3 + var4 * var5", (["var1"], ["var2 * var3", "var4 * var5"], [])),
    # Test case with only interactions
    ("y ~ a * b * c", (["y"], ["a * b * c"], [])),
    # Test case with no interactions and multiple variables
    ("y ~ a + b + c + d", (["y"], [], ["a", "b", "c", "d"])),
    # Test case with no interactions and no variables
    ("y ~ ", (["y"], [], [""])),
    # Test case with no response variable
    (" ~ a + b", ([""], [], ["a", "b"])),
])
def test_parse_model_formula_valid(formula, expected):
    assert parse_model_formula(formula) == expected

@pytest.mark.parametrize("formula, error_message", [
    # Test cases for invalid formulas
    ("y a + b", "Formula must contain one (and only one) '~' character."),
    ("y ~ a + b ~ c", "Formula must contain one (and only one) '~' character."),
    ("", "Formula must contain one (and only one) '~' character."),
])
def test_parse_model_formula_invalid(formula, error_message):
    with pytest.raises(ValueError) as excinfo:
        parse_model_formula(formula)
    assert str(excinfo.value) == error_message


# Test relevel_categorical_variable function
import pandas as pd
from stepsel.modeling.prep.helper import relevel_categorical_variable

@pytest.fixture
def create_categorical_series():
    """Fixture to create a basic categorical series for testing."""
    cat_series = pd.Series(["a", "b", "c", "a"], dtype="category")
    return cat_series

@pytest.mark.parametrize("new_order, expected_categories, expected_categories_alternative", [
    (["b", "a", "c"], ["b", "a", "c"], []),
    (["c"], ["c", "a", "b"], ["c", "b", "a"]), # or ["c", "b", "a"] ... order doesn't matter
    (["a", "c"], ["a", "c", "b"], []),
])
def test_relevel_categorical_variable_valid(create_categorical_series, new_order, expected_categories, expected_categories_alternative):
    result = relevel_categorical_variable(create_categorical_series, new_order)
    assert (list(result.cat.categories) == expected_categories) or (list(result.cat.categories) == expected_categories_alternative)

@pytest.mark.parametrize("new_order, error_message", [
    (["x", "a", "b"], "New order is not a subset of current categories:"),
    (["a", "a", "b"], "New order contains duplicates:"),
])
def test_relevel_categorical_variable_invalid(create_categorical_series, new_order, error_message):
    with pytest.raises(ValueError) as excinfo:
        relevel_categorical_variable(create_categorical_series, new_order)
    assert error_message in str(excinfo.value)

def test_relevel_categorical_variable_non_categorical():
    non_cat_series = pd.Series(["a", "b", "c", "a"])
    new_order = ["b", "a", "c"]
    result = relevel_categorical_variable(non_cat_series, new_order)
    assert list(result.cat.categories) == new_order
    assert result.dtype.name == "category"

# Test recognize_variable_types function
from stepsel.modeling.prep.helper import recognize_variable_types

@pytest.fixture
def data():
    dt = {
        "y": list(range(1, 11)),
        "var1": list(range(1, 11)),  # Numerical
        "var2": list(range(11, 21)),  # Numerical
        "cat1": pd.Categorical(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]),  # Categorical
        "cat2": pd.Categorical(["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"]),  # Categorical
        "x1": list(range(21, 31)),  # Numerical
        "x2": pd.Categorical(["C", "D", "C", "D", "C", "D", "C", "D", "C", "D"]),  # Categorical
        "break": ["U", "V", "U", "V", "U", "V", "U", "V", "U", "V"]
    }
    return pd.DataFrame(dt)

@pytest.mark.parametrize("interaction_vars, non_interaction_vars, expected", [
    ([], ['var1', 'var2', 'x1'], {'non_interaction_numerical_variables': ['var1', 'var2', 'x1'], 'non_interaction_categorical_variables': [], 'interaction_numerical_variables': [], 'interaction_categorical_variables': [], 'interaction_variables': []}),
    ([], ['cat1', 'cat2', 'x2'], {'non_interaction_numerical_variables': [], 'non_interaction_categorical_variables': ['cat1', 'cat2', 'x2'], 'interaction_numerical_variables': [], 'interaction_categorical_variables': [], 'interaction_variables': []}),
    ([], ['var1', 'cat1', 'x2'], {'non_interaction_numerical_variables': ['var1'], 'non_interaction_categorical_variables': ['cat1', 'x2'], 'interaction_numerical_variables': [], 'interaction_categorical_variables': [], 'interaction_variables': []}),
    (['var1 * var2'], [], {'non_interaction_numerical_variables': [], 'non_interaction_categorical_variables': [], 'interaction_numerical_variables': ['var2', 'var1'], 'interaction_categorical_variables': [], 'interaction_variables': ['var1 * var2']}),
    (['cat1 * cat2'], [], {'non_interaction_numerical_variables': [], 'non_interaction_categorical_variables': [], 'interaction_numerical_variables': [], 'interaction_categorical_variables': ['cat1', 'cat2'], 'interaction_variables': ['cat1 * cat2']}),
    (['var1 * cat1', 'var2 * x2'], [], {'non_interaction_numerical_variables': [], 'non_interaction_categorical_variables': [], 'interaction_numerical_variables': ['var2', 'var1'], 'interaction_categorical_variables': ['cat1', 'x2'], 'interaction_variables': ['var1 * cat1', 'var2 * x2']}),
    (['var1 * cat1'], ['var2', 'cat2'], {'non_interaction_numerical_variables': ['var2'], 'non_interaction_categorical_variables': ['cat2'], 'interaction_numerical_variables': ['var1'], 'interaction_categorical_variables': ['cat1'], 'interaction_variables': ['var1 * cat1']}),
    ([], ['var1', 'var2', 'cat1', 'cat2', 'x1', 'x2'], {'non_interaction_numerical_variables': ['var1', 'var2', 'x1'], 'non_interaction_categorical_variables': ['cat1', 'cat2', 'x2'], 'interaction_numerical_variables': [], 'interaction_categorical_variables': [], 'interaction_variables': []}),
    (['var1 * var2', 'var1 * cat1', 'cat2 * var2', 'cat1 * cat2'], ['x1', 'x2'], {'non_interaction_numerical_variables': ['x1'], 'non_interaction_categorical_variables': ['x2'], 'interaction_numerical_variables': ['var2', 'var1'], 'interaction_categorical_variables': ['cat1', 'cat1', 'cat2', 'cat2'], 'interaction_variables': ['var1 * var2', 'var1 * cat1', 'cat2 * var2', 'cat1 * cat2']}),
])
def test_recognize_variable_types(data, interaction_vars, non_interaction_vars, expected):
    assert compare_dictionaries(recognize_variable_types(data, interaction_vars, non_interaction_vars), expected), "Variable types not recognized correctly."

@pytest.mark.parametrize("interaction_vars, non_interaction_vars, error_message", [
    (["var1 * var2"], ['x1', 'break'],
     """Non-interaction variables must be either numerical or categorical.

                            Non-interaction variables: ['x1', 'break']

                            Numerical variables: ['x1']

                            Categorical variables: []"""),
    (["var1 * break"], [],
     """Interaction variables must be either numerical or categorical.

                             Interaction variables: ['var1 * break']

                             Numerical variables: ['var1']

                             Categorical variables: []""")
])
def test_recognize_variable_types_invalid(data, interaction_vars, non_interaction_vars, error_message):
    with pytest.raises(ValueError) as excinfo:
        recognize_variable_types(data, interaction_vars, non_interaction_vars)
    assert error_message in str(excinfo.value)
