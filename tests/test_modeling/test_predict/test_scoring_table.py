import pytest
import numpy as np
import pandas as pd
from stepsel.datasets import load_dummy_data
from stepsel.modeling.prep.model_matrix import prepare_model_matrix, adjust_model_matrix
from stepsel.modeling.predict.scoring_table import ScoringTableGLM
from statsmodels.api import GLM
from statsmodels.api import families

"""
sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1"],
        "var2": [None, None, None, None],
        "level_var1": [None, "A", "B", "C"],
        "level_var2": [None, None, None, None],
        "estimate": [0.5, 0, 0.2, 0.3]}

sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1"],
        "var2": [None, None, None, None, "x1", "x1", "x1"],
        "level_var1": [None, "A", "B", "C", "A", "B", "C"],
        "level_var2": [None, None, None, None, None, None, None],
        "estimate": [0.5, 0, 0.2, 0.3, 2, 3, 4]}

sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1",  "cat1", "cat1", "cat1"],
        "var2": [None, None, None, None, "x1", "x1", "x1", "cat2", "cat2", "cat2", "cat2", "cat2", "cat2"],
        "level_var1": [None, "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
        "level_var2": [None, None, None, None, None, None, None, "X", "X", "X", "Y", "Y", "Y"],
        "estimate": [0.5, 0, 0.2, 0.3, 2, 3, 4, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}

sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1",  "cat1", "cat1", "cat1", "x1"],
        "var2": [None, None, None, None, "x1", "x1", "x1", "cat2", "cat2", "cat2", "cat2", "cat2", "cat2", None],
        "level_var1": [None, "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", None],
        "level_var2": [None, None, None, None, None, None, None, "X", "X", "X", "Y", "Y", "Y", None],
        "estimate": [0.5, 0, 0.2, 0.3, 2, 3, 4, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.7]}

sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1",  "cat1", "cat1", "cat1", "x1", "x1"],
        "var2": [None, None, None, None, "x1", "x1", "x1", "cat2", "cat2", "cat2", "cat2", "cat2", "cat2", None, "x2"],
        "level_var1": [None, "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", None, None],
        "level_var2": [None, None, None, None, None, None, None, "X", "X", "X", "Y", "Y", "Y", None, None],
        "estimate": [0.5, 0, 0.2, 0.3, 2, 3, 4, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.7, 0.8]}
"""


@pytest.fixture
def sct_data():
    sct_data = {"var1": ["Intercept", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1", "cat1",  "cat1", "cat1", "cat1", "x1", "x1"],
        "var2": [None, None, None, None, "x1", "x1", "x1", "cat2", "cat2", "cat2", "cat2", "cat2", "cat2", None, "x2"],
        "level_var1": [None, "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", None, None],
        "level_var2": [None, None, None, None, None, None, None, "X", "X", "X", "Y", "Y", "Y", None, None],
        "estimate": [0.8002, 0.0000,-0.2669,-0.2381,-0.1496, 0.0902,-0.0705, 0.0000,-0.1392, 0.0420,-0.2330,-0.1277,-0.2801,-0.1299, 0.0952]}
    return sct_data

@pytest.fixture
def glm_model():
    # Load data for modeling
    dt = load_dummy_data()
    # Fit a model
    y, X, feature_ids = prepare_model_matrix("y ~ cat1 + cat1 * x1 + cat1 * cat2 + x1 + x1 * x2", dt)
    model = GLM(y, X, family=families.Gaussian()).fit()
    return model, X, y


def test_scoring_table_load_types_predictions(sct_data, glm_model):
    # Scoring table data
    sct = pd.DataFrame(sct_data)
    # GLM model
    glm_model, X, _ = glm_model

    # Create scoring table
    sct_dt = ScoringTableGLM(sct.loc[~sct.index.isin([1, 7]),:].reset_index(drop=True)) # from DataFrame
    sct_csv = ScoringTableGLM.from_csv("./src/stepsel/data/scoring_table.csv") # from CSV
    sct_glm = ScoringTableGLM.from_glm_model(glm_model) # from GLM model

    # For GLM round to 4 decimal places
    sct_glm.scoring_table["estimate"] = sct_glm.scoring_table["estimate"].round(4)

    # Get predictions
    preds_dt = sct_dt.predict_linear(X)
    preds_csv = sct_csv.predict_linear(X)
    preds_glm = sct_glm.predict_linear(X)

    # Check if predictions are the same for all loading methods
    assert np.allclose(preds_dt, preds_csv), "Predictions differ between DataFrame and CSV"
    assert np.allclose(preds_dt, preds_glm), "Predictions differ between DataFrame and GLM"
    assert np.allclose(preds_csv, preds_glm), "Predictions differ between CSV and GLM"

    # Check if SQL string is as expected
    expected_sql = " + 0.8002 + x1 * -0.1299 + case  when TRIM(CAST(cat1 as varchar(999))) = 'B' then -0.2669 when TRIM(CAST(cat1 as varchar(999))) = 'C' then -0.2381 else 0.0 end + case  when TRIM(CAST(cat1 as varchar(999))) = 'A' then x1 * -0.1496 when TRIM(CAST(cat1 as varchar(999))) = 'B' then x1 * 0.0902 when TRIM(CAST(cat1 as varchar(999))) = 'C' then x1 * -0.0705 else 0.0 end + case  when TRIM(CAST(cat1 as varchar(999))) = 'A' and TRIM(CAST(cat2 as varchar(999))) = 'Y' then -0.233 when TRIM(CAST(cat1 as varchar(999))) = 'B' and TRIM(CAST(cat2 as varchar(999))) = 'X' then -0.1392 when TRIM(CAST(cat1 as varchar(999))) = 'B' and TRIM(CAST(cat2 as varchar(999))) = 'Y' then -0.1277 when TRIM(CAST(cat1 as varchar(999))) = 'C' and TRIM(CAST(cat2 as varchar(999))) = 'X' then 0.042 when TRIM(CAST(cat1 as varchar(999))) = 'C' and TRIM(CAST(cat2 as varchar(999))) = 'Y' then -0.2801 else 0.0 end + x1 * x2 * 0.0952"
    assert sct_glm.sql() == expected_sql, "SQL string is not as expected"
    with pytest.raises(ValueError):
        sct_glm.sql(intercept_name="intercept_not_in_table")
    with pytest.raises(TypeError):
        sct_glm.sql(intercept_name=1)

    # Check type of scoring table
    assert isinstance(sct_dt, ScoringTableGLM), "Scoring table is not of type ScoringTableGLM"


def test_scoring_table_glm_with_valid_adjustments(glm_model):
    # GLM model
    _, X, y = glm_model

    X = X.loc[:, ["Intercept", "cat1: B", "cat1: C"]]
    glm_model = GLM(y, X, family=families.Gaussian()).fit()

    # Predictions before adjustments
    sct_glm = ScoringTableGLM.from_glm_model(glm_model) # from GLM model
    preds_before = sct_glm.predict_linear(X)

    # Adjustments
    adjusted_coeffs = {'cat1: B': 10}
    X_adj, offset_adj = adjust_model_matrix([X], adjusted_coeffs)

    glm_model_adj = GLM(y, X_adj, offset=offset_adj, family=families.Gaussian()).fit()

    # Create scoring table
    sct_glm_adj = ScoringTableGLM.from_glm_model(glm_model_adj, adjusted_coeffs) # from GLM model

    # Predictions after adjustments
    preds_after = sct_glm_adj.predict_linear(X)

    # Check if predictions are different before and after adjustments
    assert not np.allclose(preds_before, preds_after), "Predictions are the same before and after adjustments"


def test_scoring_table_invalid_load(sct_data):
    sct = pd.DataFrame(sct_data)
    sct.columns = ["var1", "var2", "level_variable1", "level_var2", "estimate"]
    with pytest.raises(ValueError):
        ScoringTableGLM(sct)
    with pytest.raises(TypeError):
        ScoringTableGLM(sct_data)
    with pytest.raises(ValueError):
        ScoringTableGLM.from_csv("./src/stepsel/data/scoring_table_invalid.csv")


def test_scoring_table_missing_columns(sct_data):
    sct = ScoringTableGLM(pd.DataFrame(sct_data).iloc[:4,:])
    dt = load_dummy_data()
    y, X, feature_ids = prepare_model_matrix("y ~ cat1", dt)

    # Test if missing columns raise an warning exception
    with pytest.warns(UserWarning):
        sct.predict_linear(X)


def test_scoring_table_to_sql_and_csv(sct_data):
    sct = ScoringTableGLM(pd.DataFrame(sct_data))
    with pytest.raises(ImportError):
        sct.to_sql("table_name", "schema_name")
    with pytest.raises(OSError):
        sct.to_csv("/invalid/path/to/file.csv")

