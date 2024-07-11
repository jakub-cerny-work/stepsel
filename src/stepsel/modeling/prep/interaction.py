"""
Preprocessing functions for interaction terms.
"""
import pandas as pd

def interaction_categorical_numerical(series1: pd.Series, series2: pd.Series):
    """ Create an interaction term between a categorical and a numerical variable.

    Parameters
    ----------
    series1 : pandas.Series
        The first series.

    series2 : pandas.Series
        The second series.
    
    Returns
    -------
    interaction_df : pandas.DataFrame
        A DataFrame with the interaction terms.
    
    Raises
    ------
    ValueError
        If one (and only one) of the series is not categorical.
        If one (and only one) of the series is not numerical.

    Notes
    -----
    The function will create dummy variables for the categorical variable and multiply them by the numerical variable.
    The dummy variables will be named "categorical_variable: category * numerical_variable".

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from stepsel.modeling.prep import interaction_categorical_numerical
    >>> categorical_series = pd.Series(np.random.choice(["A", "B", "C"], size=10), name="categorical").astype("category")
    >>> numerical_series = pd.Series(np.random.normal(size=10), name="numerical")
    >>> interaction_categorical_numerical(categorical_series, numerical_series)
            categorical: A * numerical  categorical: B * numerical  categorical: C * numerical
    0                        -0.626453                    0.417258                    0.619825
    1                         0.183643                   -0.720788                   -0.720788
    2                         0.835979                   -0.632650                   -0.632650
    ...
    """
    # Check if one of the series is categorical
    if (series1.dtype.name == "category") * 1 + (series2.dtype.name == "category") * 1 != 1:
        raise ValueError("One (and only one) of the series must be categorical.")
    
    # Check if one of the series is numerical
    if (series1.dtype.kind in 'biufc') * 1 + (series2.dtype.kind in 'biufc') * 1 != 1:
        raise ValueError("One (and only one) of the series must be numerical.")

    # Get categorical and numerical series
    if series1.dtype.name == "category":
        categorical_series = series1
        numerical_series = series2
    else:
        categorical_series = series2
        numerical_series = series1

    # Get interaction, create DataFrame
    categorical_dummy_df = pd.get_dummies(categorical_series, prefix=categorical_series.name, prefix_sep=": ",
                                          columns=categorical_series.name, dtype=int)

    # Multiply dummy variables by numerical variable
    for col in categorical_dummy_df.columns:
        categorical_dummy_df[col] = categorical_dummy_df[col] * numerical_series

    # Set interaction name
    categorical_dummy_df.columns = [f"{x} * {numerical_series.name}" for x in categorical_dummy_df.columns]
    
    return categorical_dummy_df


def interaction_categorical_categorical(series1: pd.Series, series2: pd.Series):
    """ Create an interaction term between two categorical variables.

    Parameters
    ----------
    series1 : pandas.Series
        The first series.

    series2 : pandas.Series
        The second series.

    Returns
    -------
    interaction : pandas.Series
        A Series with the interaction terms.

    Raises
    ------
    ValueError
        If one series1 is not categorical.
        If one series2 is not categorical.

    Notes
    -----
    The function will create an interaction term between the two categorical variables.
    The interaction term will be named "categorical_variable1 * categorical_variable2".
    The interactions will be in form of "category1 * category2".

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from stepsel.modeling.prep import interaction_categorical_categorical
    >>> categorical_series1 = pd.Series(["A", "B", "C"], name="categorical1").astype("category")
    >>> categorical_series2 = pd.Series(["X", "Y", "Z"], name="categorical2").astype("category")
    >>> interaction_categorical_categorical(categorical_series1, categorical_series2)
    0    A * X
    1    B * Y
    2    C * Z
    Name: categorical1 * categorical2, dtype: category
    Categories (3, object): ['A * X', 'B * Y', 'C * Z']   
    """
    # Check if series are categorical
    if not series1.dtype.name == "category":
        raise ValueError("Series1 is not categorical.")
    if not series2.dtype.name == "category":
        raise ValueError("Series2 is not categorical.")

    # Get categories
    series1_cats = series1.cat.categories
    series2_cats = series2.cat.categories

    # Get interaction
    interaction = series1.astype(str).str.cat(series2.astype(str), sep=' * ').astype("category")

    # Get interaction categories (first is interaction of reference categories)
    interaction_categories = [f"{x} * {y}" for x in series1_cats for y in series2_cats]
    
    # Reorder categories (for ordering I need to add new categories, and remove them afterwards, so they are not filled with 0s)
    new_categories = [x for x in interaction_categories if x not in interaction.cat.categories]
    interaction = interaction.cat.add_categories(new_categories)
    interaction = interaction.cat.reorder_categories(interaction_categories)
    interaction = interaction.cat.remove_unused_categories()

    # Set interaction name
    interaction.name = f"{series1.name} * {series2.name}"

    return interaction


def interaction_numerical_numerical(series1: pd.Series, series2: pd.Series):
    """ Create an interaction term between two numerical variables.

    Parameters
    ----------
    series1 : pandas.Series
        The first series.
        
    series2 : pandas.Series
        The second series.

    Returns
    -------
    interaction : pandas.Series
        A Series with the interaction terms.

    Raises
    ------
    ValueError
        If one series1 is not numerical.
        If one series2 is not numerical.

    Notes
    -----
    The function will create an interaction term between the two numerical variables.
    The interaction term will be named "numerical_variable1 * numerical_variable2".

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from stepsel.modeling.prep import interaction_numerical_numerical
    >>> numerical_series1 = pd.Series(np.random.normal(size=10), name="numerical1")
    >>> numerical_series2 = pd.Series(np.random.normal(size=10), name="numerical2")
    >>> interaction_numerical_numerical(numerical_series1, numerical_series2)
    0    -0.626453
    1    -0.720788
    2    -0.632650
    ...
    Name: numerical1 * numerical2, dtype: float64
    """
    # Check if both series are numerical
    if not series1.dtype.kind in 'biufc':
        raise ValueError("Series1 is not numerical.")
    if not series2.dtype.kind in 'biufc':
        raise ValueError("Series2 is not numerical.")
    
    # Multiply series
    interaction = series1 * series2

    # Set interaction name
    interaction.name = f"{series1.name} * {series2.name}"

    return interaction

