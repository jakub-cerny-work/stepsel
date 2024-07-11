import pandas as pd
import numpy as np
from stepsel.modeling.prep import (
    parse_model_formula, recognize_variable_types, get_interaction_type,
    interaction_categorical_numerical, interaction_categorical_categorical,
    interaction_numerical_numerical
)


def prepare_model_matrix(formula: str, data: pd.DataFrame, intercept: bool = True, drop_first: bool = True,
                         omit_left_side_variables: bool = False):
    """ Prepare a model matrix based on a formula and a data set.
    TODO: If intercept = False, keep all the levels of the first categorical variable.

    Parameters
    ----------
    formula : str
        The formula for the model.

    data : pandas.DataFrame
        The data set.

    intercept : bool, optional
        Whether to include an intercept in the model matrix. Default is True.

    drop_first : bool, optional
        Whether to drop the first level of each categorical variable. Default is True.

    omit_left_side_variables : bool, optional
        Whether to omit the left side variables from the output. Default is False.
        If True, the function will return only the model matrix and the feature IDs.

    Returns
    -------
    y : pandas.Series
        The response variable. If omit_left_side_variables is True, the function won't return y.

    model_matrix : pandas.DataFrame
        The model matrix.
    
    feature_ids : list
        The feature IDs.

    Raises
    ------
    ValueError
        If interaction type is not supported.

    Notes
    -----
    The function will create a model matrix based on the formula and the data set. Categories will be dummy-encoded.
    Interaction terms will be created and dummy-encoded if necessary.
    The feature IDs will be a list of strings of the variable names corresponding to the columns of the model matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from stepsel.modeling.prep import prepare_model_matrix
    >>> data = pd.DataFrame({"y": np.random.normal(size=100),
    ...                      "x1": np.random.normal(size=100),
    ...                      "x2": np.random.choice(["A", "B", "C"], size=100),
    ...                      "x3": np.random.choice(["A", "B", "C"], size=100)})
    >>> data[["x2", "x3"]] = data[["x2", "x3"]].astype("category")
    >>> y, model_matrix, feature_ids = prepare_model_matrix("y ~ x1 + x2 + x3 + x1*x2 + x1*x3", data)   
    """
    # Parse formula
    left_side_variables, interaction_variables, non_interaction_variables = parse_model_formula(formula)

    # Recognize variable types of parsed formula
    variables_by_type = recognize_variable_types(data, interaction_variables, non_interaction_variables)
    non_interaction_numerical_variables = variables_by_type["non_interaction_numerical_variables"]
    non_interaction_categorical_variables = variables_by_type["non_interaction_categorical_variables"]
    interaction_variables = variables_by_type["interaction_variables"]
    interaction_numerical_variables = variables_by_type["interaction_numerical_variables"]
    interaction_categorical_variables = variables_by_type["interaction_categorical_variables"]

    # Create model matrix: non-interaction numerical variables
    model_matrix = pd.DataFrame()
    feature_ids = []
    for variable in non_interaction_numerical_variables:
        model_matrix = pd.concat([model_matrix, data[variable]], axis=1)
        feature_ids.append(variable)
    
    # Create model matrix: non-interaction categorical variables
    for variable in non_interaction_categorical_variables:
        model_matrix = pd.concat([model_matrix, pd.get_dummies(data[variable], prefix=variable, prefix_sep=": ",
                                                               columns=data[variable].name, drop_first=drop_first, dtype=int)],
                                    axis=1)
        feature_ids.extend([variable] * (model_matrix.shape[1] - len(feature_ids)))
    
    # Create model matrix: interaction variables
    for interaction in interaction_variables:
        interaction_split = interaction.split("*")
        interaction_split = [x.strip() for x in interaction_split]
        # Type of interaction
        interaction_type = get_interaction_type(interaction, interaction_numerical_variables, interaction_categorical_variables)
        if interaction_type in ["numerical_categorical", "categorical_numerical"]:
            series1 = data[interaction_split[0]]
            series2 = data[interaction_split[1]]
            model_matrix = pd.concat([model_matrix, interaction_categorical_numerical(series1, series2)],
                                      axis=1)
            feature_ids.extend([interaction] * (model_matrix.shape[1] - len(feature_ids)))
        elif interaction_type == "categorical_categorical":
            series1 = data[interaction_split[0]]
            series2 = data[interaction_split[1]]
            series_interaction = interaction_categorical_categorical(series1, series2)
            model_matrix = pd.concat([model_matrix, pd.get_dummies(series_interaction, prefix=series_interaction.name, prefix_sep=": ",
                                                                    columns=series_interaction.name, drop_first=drop_first, dtype=int)],
                                      axis=1)
            feature_ids.extend([interaction] * (model_matrix.shape[1] - len(feature_ids)))
        elif interaction_type == "numerical_numerical":
            series1 = data[interaction_split[0]]
            series2 = data[interaction_split[1]]
            model_matrix = pd.concat([model_matrix, interaction_numerical_numerical(series1, series2)],
                                      axis=1)
            feature_ids.extend([interaction] * (model_matrix.shape[1] - len(feature_ids)))
        else:
            raise ValueError(f"""Interaction type not recognized.
                                 Interaction: {interaction}
                                 Interaction type: {interaction_type}
                                 """)
        
    # Add intercept
    if intercept:
        model_matrix = pd.concat([pd.Series([1] * model_matrix.shape[0], name="Intercept"), model_matrix], axis=1)
        feature_ids.insert(0, "Intercept")

    # Target variable
    if omit_left_side_variables:
        return model_matrix, feature_ids
    else:
        y = data[left_side_variables]
        return y, model_matrix, feature_ids


def adjust_model_matrix(model_matrices: list, adjusted_coeffs: dict, offsets: list = None):
    """ Adjust model matrix (and offset) based on adjusted coefficients dictionary.

    Parameters
    ----------
    model_matrices : list (of data frames)
        The model matrices.
    
    adjusted_coeffs : dict
        The adjusted coefficients dictionary.
        The format of the dictionary is as follows:
            {variable_name: adjusted_coefficient}
            Variable_name is the name of the variable in the model.
            Example: {"ts_new9_g: 06": 0.20, "drpou_cpp_dop3: H": -1.74}

    offsets : list (of numpy arrays or pandas Series), optional
        The offsets. Default is None.

    Returns
    -------
    model_matrices : tuple (of data frames)
        The adjusted model matrices.

    offsets : tuple (of numpy arrays or pandas Series)
        The adjusted offsets.

    Raises
    ------
    Exception
        If the number of offsets is not equal to the number of model matrices.
        If the number of rows in the model matrix is not equal to the number of offset values.

    Notes
    -----
    The function will adjust the model matrices and offsets based on the adjusted coefficients dictionary.
    The function will delete the variables from the model matrices and add the adjusted coefficients to the offsets.
    The function will return a tuple of the adjusted model matrices and offsets.
    Adjustments are done in-place. If both matrices and offsets are provided, re-assignment is not necessary.
    If one wants to keep the original model matrices and offsets, make a copy of them before calling the function.
    """
    # Create offsets if not provided
    if offsets is None:
        offsets = [None] * len(model_matrices)
        for i in range(len(model_matrices)):
            offsets[i] = np.zeros(model_matrices[i].shape[0])

    # Check if the number of offsets is equal to the number of model matrices
    elif len(offsets) != len(model_matrices):
        raise Exception("The number of offsets must be equal to the number of model matrices.")
    
    # Check if the shape of the offsets is equal to the number of rows in the model matrices
    else:
        for i in range(len(offsets)):
            if offsets[i].shape[0] != model_matrices[i].shape[0]:
                raise Exception(f"The number of rows in the model matrix {i} is not equal to the number of offset values.")
    
    # Adjust the model matrices and offsets
    for k, v in adjusted_coeffs.items():
        for i in range(len(model_matrices)):
            offsets[i] += (v * model_matrices[i][k])
            model_matrices[i].drop(k, axis=1, inplace=True)

    return tuple(model_matrices + offsets)

