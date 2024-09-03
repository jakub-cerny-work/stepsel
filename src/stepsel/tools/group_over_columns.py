import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

def group_over_columns(dt: pd.DataFrame, columns: ArrayLike, agg_dict: dict) -> pd.DataFrame:
    """ Group over columns and aggregate by agg_dict

    Parameters
    ----------
    dt : pd.DataFrame
        Dataframe to group over

    columns : ArrayLike
        Columns to group over. It can be a list of columns or a list of lists of columns.

    agg_dict : dict
        Dictionary of columns and aggregation functions

    Returns
    -------
    comparison : pd.DataFrame
        Dataframe with grouped and aggregated data


    Examples
    --------
    >>> group_over_columns(dt, ["catvar1", "catvar2"], {"smlr": "sum", "preds": "mean", target: "mean"})
    >>> group_over_columns(dt, [["catvar1", "catvar2"]], {"smlr": "sum", "preds": "mean", target: "mean"})
    >>> group_over_columns(dt, [["catvar1", "catvar2"], "catvar3"], {"smlr": "sum", "preds": "mean", target: "mean"})
    """
    comparison = pd.DataFrame()
    rename_dict = {}
    for var in columns:

        # Prepare rename dict
        if np.size(var) > 1:
            for i, v in enumerate(var):
                rename_dict.update({v: f"level_{i + 1}"})
        else:
            rename_dict.update({var: "level_1"})

        # Group data
        gr = dt.groupby(var, observed=False).agg(agg_dict) #.reset_index().rename(columns=rename_dict)
        
        ## Get rid of multiindex
        if gr.columns.nlevels > 1:
            gr.columns = ["_".join(col) if col[1] != "" else col[0] for col in gr.columns.values]
            rename_dict_cols = {}
            rename_dict_cols_reversed = {}
        else:
            cols_original = gr.columns
            rename_dict_cols = {col: f"{col}_tmp" for col in cols_original} # Rename columns to avoid overwriting and error of same name
            rename_dict_cols_reversed = {f"{col}_tmp": col for col in cols_original}
        gr = gr.rename(columns=rename_dict_cols) \
                .reset_index() \
                .rename(columns=rename_dict) \
                .rename(columns=rename_dict_cols_reversed)

        # Add variable name
        if np.size(var) > 1:
            for i, v in enumerate(var):
                gr.insert(i, f"variable_{i + 1}", v)
        else:
            gr.insert(0, "variable_1", var)

        # Append to comparison
        comparison = comparison._append(gr)

    # Reset index
    comparison.reset_index(drop=True, inplace=True)

    # Reorder columns - variables and levels first, then the rest
    columns_output = list(comparison.columns)
    columns_to_move = ["variable", "level"]
    ## Number of variables
    n_variables = max(list(map(lambda x: len(x) if isinstance(x,list) else 1, columns)))
    columns_to_move = sum([list(map(lambda x: f"{x}_{i + 1}", columns_to_move)) for i in np.arange(n_variables)], [])
    # Keeps order of aggregated columns
    for col in columns_output:
        if col not in columns_to_move:
            columns_to_move.append(col)

    comparison = comparison[columns_to_move]

    return comparison
