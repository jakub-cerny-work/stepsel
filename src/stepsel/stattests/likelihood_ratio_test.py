""" Likelihood ratio test for nested models.

References
# https://api.rpubs.com/tomanderson_34/lrt
# https://towardsdatascience.com/the-likelihood-ratio-test-463455b34de9
"""
from scipy.stats import chi2
import statsmodels.api as sm

def likelihood_ratio_test(llf_complex, llf_nested, df_complex, df_nested):
    """ Calculate likelihood ratio test for nested models.

    Parameters
    ----------
    llf_complex : float
        Log-likelihood of complex model.
    llf_nested : float
        Log-likelihood of nested model.
    df_complex : int
        Degrees of freedom of complex model.
    df_nested : int
        Degrees of freedom of nested model.

    Returns
    -------
    tuple
        lr : float
            Likelihood ratio test statistic.
        p : float
            p-value of likelihood ratio test.
    """
    # Test statistic
    lr = -2 * (llf_nested - llf_complex)
    # p-value
    p = chi2.sf(lr, df_complex - df_nested)

    return lr, p


def likelihood_ratio_test_models(complex: sm.GLM, nested: sm.GLM):
    """ Calculate likelihood ratio test for nested models.

    Parameters
    ----------
    complex : sm.GLM
        Complex model.
    nested : sm.GLM
        Nested model.

    Returns
    -------
    tuple
        lr : float
            Likelihood ratio test statistic.
        p : float
            p-value of likelihood ratio test.
    """
    # Get log-likelihood
    llf_complex = complex.llf
    llf_nested = nested.llf
    # Get degrees of freedom
    df_complex = complex.df_model
    df_nested = nested.df_model

    lr, p = likelihood_ratio_test(llf_complex, llf_nested, df_complex, df_nested)

    return lr, p
