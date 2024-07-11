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
    >>> group_over_columns(dt, ["ts_new9_g", "drzitel_vek_nace_kat2"], {"smlr": "sum", "preds": "mean", target: "mean"})
    >>> group_over_columns(dt, [["ts_new9_g", "drzitel_vek_nace_kat2"]], {"smlr": "sum", "preds": "mean", target: "mean"})
    >>> group_over_columns(dt, [["ts_new9_g", "drzitel_vek_nace_kat2"], "drpou_cpp_dop3"], {"smlr": "sum", "preds": "mean", target: "mean"})
    """
    comparison = pd.DataFrame()
    for var in columns:

        # Prepare rename dict
        if np.size(var) > 1:
            rename_dict = {}
            for i, v in enumerate(var):
                rename_dict.update({v: f"level_{i + 1}"})
        else:
            rename_dict = {var: "level_1"}

        # Group data
        gr = dt.groupby(var).agg(agg_dict).reset_index().rename(columns=rename_dict)

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

    # Get rid of multiindex
    if comparison.columns.nlevels > 1:
        comparison.columns = ["_".join(col) if col[1] != "" else col[0] for col in comparison.columns.values]

    # Reorder columns - variables and levels first, then the rest
    if comparison.shape[1] == (len(agg_dict) + 2):
        # Only one variable - no need to reorder
        # Rename to be consistent with previous version output
        comparison = comparison.rename(columns={"variable_1": "variable", "level_1": "level"})
    else:
        columns = list(comparison.columns)
        columns_to_move = ["variable", "level"]
        columns_to_move = sum([list(map(lambda x: f"{x}_{i + 1}", columns_to_move)) for i in np.arange(np.size(var))], [])
        # Keeps order of aggregated columns
        for col in columns:
            if col not in columns_to_move:
                columns_to_move.append(col)

        comparison = comparison[columns_to_move]

    return comparison
