import pytest
import numpy as np
import pandas as pd
from stepsel.modeling.prep import prepare_model_matrix
from stepsel.modeling import StepwiseGLM
from stepsel.datasets import load_soccer_data
import statsmodels.api as sm

@pytest.fixture
def data():
    # Load data
    dt = load_soccer_data()
    # Convert to categorical
    dt[['team','hga','team_opp']] = dt[['team','hga','team_opp']].astype('category')
    return dt

@pytest.mark.parametrize("test_no, formula, include, slentry, slstay, expected", [
    (1, "goals ~ team + hga + team_opp", None, 0.1, 0.2, {"Df Model": 31, "Intercept": 0.1515}),
    (2, "goals ~ team + hga + team_opp", ['penalty'], 0.1, 0.2, {"Df Model": 32, "Intercept": 0.1429}),
    (3, "goals ~ team + hga + team_opp + attacks", None, 0.1, 0.01, {"Df Model": 31, "Intercept": 0.1515}), # Cycling stepwise
    (4, "goals ~ team + hga + team_opp", ['penalty'], 0.1, 0.2, {"Df Model": 32, "Intercept": 0.1429, "Df Model 2": 33, "Intercept 2": 0.5872}),
])
def test_stepwise_01(data, test_no, formula, include, slentry, slstay, expected):

    model1 = StepwiseGLM(formula = formula,
                        data = data,
                        include = include,
                        slentry = slentry,
                        slstay = slstay,
                        family = sm.families.Poisson())
    model1.fit()
    
    actual_df_model = model1.current_model.df_model
    actual_intercept = np.round(model1.current_model.params['Intercept'], 4)

    assert actual_df_model == expected["Df Model"]
    assert actual_intercept == expected["Intercept"]

    if test_no == 4:
        formula_new = "goals ~ team + hga + team_opp + attacks"
        model2 = StepwiseGLM(formula = formula_new,
                            data = data,
                            include = include,
                            slentry = slentry,
                            slstay = slstay,
                            model_fit_log = model1.model_fit_log,
                            family = sm.families.Poisson())
        model2.fit()
        
        actual_df_model_new = model2.current_model.df_model
        actual_intercept_new = np.round(model2.current_model.params['Intercept'], 4)

        assert actual_df_model_new == expected["Df Model 2"]
        assert actual_intercept_new == expected["Intercept 2"]
