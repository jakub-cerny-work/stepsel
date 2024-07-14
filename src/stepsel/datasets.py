import pandas as pd
from importlib import resources

def load_soccer_data():
    """
    Load the soccer data from a parquet file included in the package.

    Returns
    -------
    pandas.DataFrame
        The loaded soccer data.
    """
    # Use 'importlib.resources' to ensure the path is correct after installation
    data_path = resources.files("stepsel.data").joinpath("soccer.parquet")
    dt = pd.read_parquet(data_path)    
    
    return dt
