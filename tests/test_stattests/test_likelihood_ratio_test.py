import pytest

# Test likelihood_ratio_test function
import numpy as np
from scipy.stats import chi2
from stepsel.stattests import likelihood_ratio_test

@pytest.mark.parametrize("llf_complex, llf_nested, df_complex, df_nested, expected_lr, expected_p", [
    (-10, -20, 5, 3, 20, chi2.sf(20, 2)),
    (-1675.8, -1678.7, 4, 3, 5.800000000000182, 0.016026174547983465),
    (-15, -30, 10, 8, 30, chi2.sf(30, 2)),
    (-1070.2, -1666.2, 5, 3, 1192, np.float64(1.4470674864825245e-259)),
])
def test_likelihood_ratio_test(llf_complex, llf_nested, df_complex, df_nested, expected_lr, expected_p):
    actual_lr, actual_p = likelihood_ratio_test(llf_complex, llf_nested, df_complex, df_nested)
    assert actual_lr == pytest.approx(expected_lr), "LR test statistic does not match expected value."
    assert actual_p == pytest.approx(expected_p), "P-value does not match expected value."


# Test likelihood_ratio_test_models function
import statsmodels.api as sm
from stepsel.stattests import likelihood_ratio_test_models
from stepsel.datasets import load_soccer_data

@pytest.fixture
def data():
    return load_soccer_data()

@pytest.fixture
def complex_model(data):
    model_complex = sm.formula.glm(
        formula="goals ~ team + team_opp + hga",
        data=data,
        family=sm.families.Poisson(link = sm.families.links.Log())
    )
    return model_complex.fit()

@pytest.fixture
def nested_model(data):
    model_nested = sm.formula.glm(
        formula="goals ~ team + hga",
        data=data,
        family=sm.families.Poisson(link = sm.families.links.Log())
    )
    return model_nested.fit()

def test_likelihood_ratio_test_models(complex_model, nested_model):
    lr, p = likelihood_ratio_test_models(complex_model, nested_model)
    assert lr > 0, "Likelihood ratio should be positive"
    assert 0 <= p <= 1, "P-value should be between 0 and 1"
