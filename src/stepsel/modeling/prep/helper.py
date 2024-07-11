"""
Helper functions for modeling prep. Smaller functions that are used in multiple functions.
"""
import pandas as pd

def get_interaction_type(interaction: str, interaction_numerical_variables: list, interaction_categorical_variables: list):
    """Get the interaction type of an interaction.

    Parameters
    ----------
    interaction : str
        The interaction to get the type of.
    
    interaction_numerical_variables : list
        The numerical variables that are used in the interactions.
    
    interaction_categorical_variables : list
        The categorical variables that are used in the interactions.
    
    Returns
    -------
    interaction_type : str
        The interaction type. One of "numerical_numerical", "categorical_categorical", "numerical_categorical", "categorical_numerical".

    Raises
    ------
    ValueError
        If the interaction does not contain exactly one '*' character.
        If the interaction variables are not in exactly one of the two lists.
    
    Examples
    --------
    >>> get_interaction_type("a * b", ["a"], ["b"])
    "numerical_categorical"
    """
    # Split the interaction into the two variables
    interaction_split = interaction.split("*")
    interaction_split = [x.strip() for x in interaction_split]
    if len(interaction_split) != 2:
        raise ValueError(f"Interaction must contain one (and only one) '*' character. Interaction: {interaction}")
    
    # Return the interaction type
    if interaction_split[0] in interaction_numerical_variables and interaction_split[1] in interaction_numerical_variables:
        return "numerical_numerical"
    elif interaction_split[0] in interaction_categorical_variables and interaction_split[1] in interaction_categorical_variables:
        return "categorical_categorical"
    elif interaction_split[0] in interaction_numerical_variables and interaction_split[1] in interaction_categorical_variables:
        return "numerical_categorical"
    elif interaction_split[0] in interaction_categorical_variables and interaction_split[1] in interaction_numerical_variables:
        return "categorical_numerical"
    else:
        raise ValueError(f"""Interaction variables must be either numerical or categorical.\n
                             Interaction: {interaction}\n
                             Numerical variables: {interaction_numerical_variables}\n
                             Categorical variables: {interaction_categorical_variables}\n
                          """)


def relevel_categorical_variable(series: pd.Series, new_order: list):
    """Relevel a categorical variable.

    Parameters
    ----------
    series : pd.Series
        The categorical variable to relevel.
    
    new_order : list
        The new order of the categories.
    
    Returns
    -------
    series : pd.Series
        The relevelled categorical variable.
    
    Raises
    ------
    ValueError
        If the new order is not a subset of the current categories.
        If the new order contains duplicates.
    """
    # Convert to categorical if not categorical
    if not series.dtype.name == "category":
        print("Series is not categorical, converting to categorical.")
        series = series.astype("category")
       
    # Check if new_order is a subset of current categories
    categories = set(series.cat.categories)
    if not set(new_order).issubset(categories):
        raise ValueError(f"New order is not a subset of current categories: {categories}")
    
    # Chek for duplicates in new_order
    if len(new_order) != len(set(new_order)):
        raise ValueError(f"New order contains duplicates: {new_order}")

    # Reorder categories
    untouched_categories = categories.copy()
    [untouched_categories.remove(x) for x in new_order]
    untouched_categories = list(untouched_categories)
    new_order = new_order + untouched_categories
    series = series.cat.reorder_categories(new_order)

    return series


def parse_model_formula(formula: str):
    """ Parse a model formula into its components.

    Parameters
    ----------
    formula : str
        The model formula to parse.

    Returns
    -------
    left_side_variables : list
        The variables on the left side of the formula.

    interaction_variables : list
        The interaction variables on the right side of the formula.

    non_interaction_variables : list
        The non-interaction variables on the right side of the formula.

    Raises
    ------
    ValueError
        If the formula does not contain exactly one '~' character.

    Examples
    --------
    >>> parse_model_formula("y ~ a + b + a * b")
    (["y"], ["a * b"], ["a", "b"])
    """
    # Split formula into left and right side
    formula_split = formula.split("~")
    if len(formula_split) != 2:
        raise ValueError("Formula must contain one (and only one) '~' character.")
    left_side = formula_split[0]
    right_side = formula_split[1]

    # Get left side variables
    left_side_variables = left_side.split("+")
    left_side_variables = [x.strip() for x in left_side_variables]

    # Get right side variables
    right_side_variables = right_side.split("+")
    right_side_variables = [x.strip() for x in right_side_variables]

    # Get interaction variables
    interaction_variables = [x for x in right_side_variables if "*" in x]
    interaction_variables = [x.strip() for x in interaction_variables]

    # Get non-interaction variables
    non_interaction_variables = [x for x in right_side_variables if "*" not in x]
    non_interaction_variables = [x.strip() for x in non_interaction_variables]

    return left_side_variables, interaction_variables, non_interaction_variables


def recognize_variable_types(data: pd.DataFrame, interaction_variables: list, non_interaction_variables: list):
    """ Recognize the types of the variables.

    Parameters
    ----------
    data : pd.DataFrame
        The data to recognize the variable types from.

    interaction_variables : list
        The interaction variables to recognize the types from.
    
    non_interaction_variables : list
        The non-interaction variables to recognize the types from.

    Returns
    -------
    dictionary : dict
        A dictionary containing the variable types.

        interaction_numerical_variables : list
            The numerical variables in the interaction variables.

        interaction_categorical_variables : list
            The categorical variables in the interaction variables.

        non_interaction_numerical_variables : list
            The numerical variables in the non-interaction variables.

        non_interaction_categorical_variables : list
            The categorical variables in the non-interaction variables.

        interaction_variables : list
            The interaction variables.

    Raises
    ------
    ValueError
        If the interaction variables are not either numerical or categorical.
        If the non-interaction variables are not either numerical or categorical.

    Examples
    --------
    >>> recognize_variable_types(data, ["a * b"], ["a", "b", "c"])
    (["a"], ["b"], [], [], ["a * b"])
    {"non_interaction_numerical_variables": ["a"],
     "non_interaction_categorical_variables": ["b"],
     "interaction_numerical_variables": ["a"],
     "interaction_categorical_variables": ["b", "c"],
     "interaction_variables": ["a * b"]}
    """
    # Get non-interaction numerical variables
    non_interaction_numerical_variables = []
    for variable in non_interaction_variables:
        if data[variable].dtype.kind in 'biufc':
            non_interaction_numerical_variables.append(variable)
    
    # Get non-interaction categorical variables
    non_interaction_categorical_variables = []
    for variable in non_interaction_variables:
        if data[variable].dtype.name == "category":
            non_interaction_categorical_variables.append(variable)

    # Check if non-iteraction numerical and categorical variables match the original non-iteraction variables
    if len(non_interaction_numerical_variables) + len(non_interaction_categorical_variables) != len(non_interaction_variables):
        raise ValueError(f"""Non-interaction variables must be either numerical or categorical.\n
                            Non-interaction variables: {non_interaction_variables}\n
                            Numerical variables: {non_interaction_numerical_variables}\n
                            Categorical variables: {non_interaction_categorical_variables}\n
                            """)

    # Get all variables in the interactions
    interaction_variables_split = []
    for interaction_variable in interaction_variables:
        interaction_variable_split = interaction_variable.split("*")
        interaction_variables_split.extend(interaction_variable_split)
    interaction_variables_split = list(set(interaction_variables_split))
    interaction_variables_split = [x.strip() for x in interaction_variables_split]

    # Get interaction numerical variables
    interaction_numerical_variables = []
    for variable in interaction_variables_split:
        if data[variable].dtype.kind in 'biufc':
            interaction_numerical_variables.append(variable)
        
    # Get interaction categorical variables
    interaction_categorical_variables = []
    for variable in interaction_variables_split:
        if data[variable].dtype.name == "category":
            interaction_categorical_variables.append(variable)

    # Check if interaction numerical and categorical variables match the original interaction variables
    if len(interaction_numerical_variables) + len(interaction_categorical_variables) != len(interaction_variables_split):
        raise ValueError(f"""Interaction variables must be either numerical or categorical.\n
                             Interaction variables: {interaction_variables}\n
                             Numerical variables: {interaction_numerical_variables}\n
                             Categorical variables: {interaction_categorical_variables}\n
                          """)

    return {"non_interaction_numerical_variables": non_interaction_numerical_variables,
            "non_interaction_categorical_variables": non_interaction_categorical_variables,
            "interaction_numerical_variables": interaction_numerical_variables,
            "interaction_categorical_variables": interaction_categorical_variables,
            "interaction_variables": interaction_variables
            }

