import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import warnings

class ScoringTableGLM():
    """ Class for scoring table and related functions

    Methods
    -------
    from_glm_model(model: GLMResultsWrapper, adjusted_coeffs: dict = None)
        Create a scoring table from the model fit. Suitable for statsmodels GLM.

    from_csv(path: str, *args, **kwargs)
        Create a scoring table from a csv file.

    sql(format: bool = False)
        Returns SQL statement for scoring table

    predict_linear(X: pd.DataFrame)
        Predict linear part of the GLM model based on the scoring table and model matrix.

    to_sql(*args, **kwargs)
        Upload scoring table to the database

    to_csv(path: str, *args, **kwargs)
        Save scoring table to a csv file

    _reconstruct_model_matrix_columns(scoring_table: pd.DataFrame)
        Reconstruct model matrix columns and model variables based on the scoring table  
    """

    def __init__(self, scoring_table: pd.DataFrame) -> None:
        """ Initialize ScoringTable object

        Parameters
        ----------
        scoring_table : pd.DataFrame
            Required columns: var1, var2, level_var1, level_var2, estimate

        Raises
        ------
        TypeError
            If scoring_table is not a pandas.DataFrame.
        ValueError
            If scoring_table does not contain the required columns.
        ValueError
            If the type of the row in the scoring table is not recognized.

        Notes
        -----
        The scoring table is a DataFrame with the following columns:
            var1: Name of the first variable in the interaction
            var2: Name of the second variable in the interaction
            level_var1: Level of the first variable in the interaction
            level_var2: Level of the second variable in the interaction
            estimate: Estimate of the interaction
        """
        # Check if scoring table is a DataFrame with the required columns
        if not isinstance(scoring_table, pd.DataFrame):
            raise TypeError("scoring_table must be a pandas.DataFrame.")
        required_columns = ["var1", "var2", "level_var1", "level_var2", "estimate"]
        if not all([x in scoring_table.columns for x in required_columns]):
            raise ValueError("scoring_table must contain the following columns: "
                             f"{', '.join(required_columns)}.")
        
        # Initialize attributes
        self.scoring_table = scoring_table
        self.model_matrix_columns, self.model_features, self.model_variables = self._reconstruct_model_matrix_columns(scoring_table)

    @classmethod
    def from_glm_model(cls, model: GLMResultsWrapper, adjusted_coeffs: dict = None) -> "ScoringTableGLM":
        """ Create a scoring table from the model fit. Suitable for statsmodels GLM.

        Parameters
        ----------
        model : GLMResultsWrapper
            Model fit of statsmodels GLM.
        
        adjusted_coeffs : dict, optional
            Dictionary of adjusted coefficients. The default is None.
            The format of the dictionary is as follows:
                {variable_name: adjusted_coefficient}
                Variable_name is the name of the variable in the model.
            Example: {"ts_new9_g: 06": 0.20, "drpou_cpp_dop3: H": -1.74}

        Returns
        -------
        ScoringTable
            Scoring table as a ScoringTable object.

        Notes
        -----
        Output of this function can be used to create scoring SQL query when uploaded to the database.
        """
        # Extract variables and coefficients from the model
        vars = model.params.index
        coeffs = model._results.params

        # If adjusted coefficients are provided, append them to the list of variables and coefficients
        if adjusted_coeffs is not None:
            # Append adjusted coefficients to the list of variables
            vars = list(vars)
            vars.extend(list(adjusted_coeffs.keys()))
            # Append adjusted coefficients to the list of coefficients
            coeffs = list(coeffs)
            coeffs.extend(list(adjusted_coeffs.values()))

        # Initialize lists to store data for DataFrame
        var1_list = []
        var2_list = []
        level1_list = []
        level2_list = []

        for var in vars:
            # Interaction with categorical variable
            if ":" in var:
                variables, levels = var.split(':')

                # Two categorical variables
                if ("*" in variables) & ("*" in levels):
                    v1, v2 = variables.strip().split('*')
                    l1, l2 = levels.strip().split('*')
                    variables = [v1.strip(), v2.strip()]
                    levels = [l1.strip(), l2.strip()]
                # One categorical and one numerical variable (categorical first)
                elif ("*" not in variables) & ("*" in levels):
                    levels, v = levels.strip().split('*')
                    variables = [variables.strip()]
                    variables.append(v.strip())
                    levels = [levels.strip(), None]
                # One categorical variable
                else:
                    variables = [variables.strip(), None]
                    levels = [levels.strip(), None]
            # Interaction of two numerical variables
            elif "*" in var:
                variables = var.strip().split('*')
                variables = [v.strip() for v in variables]
                levels = [None, None]

            # One numerical variable
            else:
                variables = [var.strip(), None]
                levels = [None, None]

            # Append the extracted information to the respective lists
            var1_list.append(variables[0])
            var2_list.append(variables[1])
            level1_list.append(levels[0])
            level2_list.append(levels[1])

        # Create a DataFrame from the lists
        data = {
            'var1': var1_list,
            'var2': var2_list,
            'level_var1': level1_list,
            'level_var2': level2_list,
            'estimate': coeffs
        }
        df = pd.DataFrame(data)

        # Sort the DataFrame if adjusted coefficients are provided.
        if adjusted_coeffs is not None:
            # Keep Intercept at the top if it is present (should be on the first row)
            if df.loc[0, "var1"] == "Intercept":
                first_row = df.iloc[:1].copy()
                sorted_rows = df.iloc[1:].sort_values(by=['var1','var2','level_var1','level_var2'], ascending=False)
                sorted_df = pd.concat([first_row, sorted_rows], axis = 0, ignore_index=True)
                df = sorted_df.copy()

        return cls(df.reset_index(drop=True))

    @classmethod
    def from_csv(cls, path: str, *args, **kwargs) -> "ScoringTableGLM":
        """ Create a scoring table from a csv file.

        Parameters
        ----------
        path : str
            Path to the csv file.

        *args, **kwargs
            Additional arguments passed to pandas.read_csv() function.

        Returns
        -------
        ScoringTable
            Scoring table as a ScoringTable object.

        Raises
        ------
        ValueError
            If the csv file does not contain the required columns.
        """
        # Read csv file
        scoring_table = pd.read_csv(path, *args, **kwargs)
        # Check if scoring table DataFrame has the required columns
        required_columns = ["var1", "var2", "level_var1", "level_var2", "estimate"]
        if not all([x in scoring_table.columns for x in required_columns]):
            raise ValueError("scoring_table must contain the following columns: "
                             f"{', '.join(required_columns)}.")
        # Convert to string where not None
        for column in ['var1', 'var2', 'level_var1', 'level_var2']:
            scoring_table[column] = scoring_table[column].map(lambda x: str(x) if x is not np.nan else None)
        return cls(scoring_table)

    def __repr__(self) -> str:
        """ Returns string representation of the ScoringTable object

        Returns
        -------
        str
            String representation of the ScoringTable object
        """
        return f"ScoringTableGLM(scoring_table={self.scoring_table})"

    @staticmethod
    def _reconstruct_model_matrix_columns(scoring_table) -> tuple:
        """ Reconstruct model matrix columns and model features and variables based on the scoring table

        Parameters
        ----------
        scoring_table : pd.DataFrame
            Required columns: var1, var2, level_var1, level_var2

        Returns
        -------
        tuple
            Tuple of two lists and dict.
            List of model matrix column names.
            List of model features.
            Dict of model variables with keys "numerical" and "categorical".

        Raises
        ------
        ValueError
            If the type of the row in the scoring table is not recognized.
        """
        # Initialize list of model matrix columns
        model_matrix_columns = []
        model_features = []
        model_variables = {"numerical": [],
                           "categorical": []}
        for index, row in scoring_table.iterrows():
            # Numerical variable (filled: var1; missing: var2, level_var1, level_var2)
            if (row["var1"] != None) & (row["var2"] == None) & (row["level_var1"] == None) & (row["level_var2"] == None):
                model_matrix_columns.append(row["var1"])
                model_features.append(row["var1"])
                model_variables["numerical"].append(row["var1"])
            # Categorical variable (filled: var1, level_var1; missing: var2, level_var2)
            elif (row["var1"] != None) & (row["var2"] == None) & (row["level_var1"] != None) & (row["level_var2"] == None):
                model_matrix_columns.append(row["var1"] + ": " + str(row["level_var1"]))
                model_features.append(row["var1"])
                model_variables["categorical"].append(row["var1"])
            # Interaction of categorical variables (filled: var1, level_var1, var2, level_var2; missing: None)
            elif (row["var1"] != None) & (row["var2"] != None) & (row["level_var1"] != None) & (row["level_var2"] != None):
                model_matrix_columns.append(row["var1"] + " * " + row["var2"] + ": " + str(row["level_var1"]) + " * " + str(row["level_var2"]))
                model_features.append(row["var1"] + " * " + row["var2"])
                model_variables["categorical"].append(row["var1"])
                model_variables["categorical"].append(row["var2"])
            # Interaction of numerical and categorical variables (filled: var1, level_var1, var2; missing: level_var2)
            elif (row["var1"] != None) & (row["var2"] != None) & (row["level_var1"] != None) & (row["level_var2"] == None):
                model_matrix_columns.append(row["var1"] + ": " + str(row["level_var1"]) + " * " + row["var2"])
                model_features.append(row["var1"] + " * " + row["var2"])
                model_variables["categorical"].append(row["var1"])
                model_variables["numerical"].append(row["var2"])
            # Interaction of numerical variables (filled: var1, var2; missing: level_var1, level_var2)
            elif (row["var1"] != None) & (row["var2"] != None) & (row["level_var1"] == None) & (row["level_var2"] == None):
                model_matrix_columns.append(row["var1"] + " * " + row["var2"])
                model_features.append(row["var1"] + " * " + row["var2"])
                model_variables["numerical"].append(row["var1"])
                model_variables["numerical"].append(row["var2"])
            # Else raise an error (should not happen)
            else:
                raise ValueError(f"""Unexpected row in the scoring table. Unable to recognize type of the row.
                                Please check if for example all the interactions are filled in correctly.
                                var1: {row["var1"]}
                                var2: {row["var2"]}
                                level_var1: {row["level_var1"]}
                                level_var2: {row["level_var2"]}""")
                    
            # Keep only unique values of model variables and features
            model_variables["numerical"] = list(set(model_variables["numerical"]))
            model_variables["categorical"] = list(set(model_variables["categorical"]))
            model_features = list(set(model_features))

        return model_matrix_columns, model_features, model_variables

    def sql(self, intercept_name: str = "Intercept", format: bool = False):
        """ Returns SQL statement for scoring table

        Parameters
        ----------
        intercept_name : str, optional
            Name of the intercept variable, by default 'Intercept'

        format : bool, optional
            If True, the SQL statement is formatted for better readability, by default False

        Returns
        -------
        str
            SQL statement for scoring table

        Uses
        ----
        self.scoring_table : pandas.DataFrame
            Scoring table with columns: var1, var2, level_var1, level_var2, estimate.

        Notes
        -----
        If format=True line breaks are added after each row. This is useful for manual inspection of the SQL statement.
        print() has to be used to print the SQL statement to the console with line breaks.
        """
        # Get attributes
        scoring_table = self.scoring_table.copy()

        # Check if intercept_name is a string
        if intercept_name is not None and not isinstance(intercept_name, str):
            raise TypeError("intercept_name must be a string.")
        
        # Check if intercept_name is in scoring_table
        if intercept_name is not None and intercept_name not in scoring_table["var1"].unique() and intercept_name not in scoring_table["var2"].unique():
            raise ValueError("intercept_name must be in scoring_table.")

        # Create w02_case DataFrame
        def get_row_sql_statement(row: pd.Series):
            """ Returns SQL statement for one row of scoring table

            Parameters
            ----------
            row : pandas.Series
                One row of scoring table

            Returns
            -------
            str
                SQL statement for one row of scoring table        
            """
            ## Intercept
            if (row['var1'] == intercept_name and pd.isnull(row['var2'])) or (row['var2'] == intercept_name and pd.isnull(row['var1'])):
                return f" + {row['estimate']}"
            ## One categorical variable
            # In var1
            elif pd.notnull(row['level_var1']) and pd.isnull(row['level_var2']) and pd.isnull(row['var2']):
                return f" when TRIM(CAST({row['var1']} as varchar(999))) = '{row['level_var1']}' then {row['estimate']}"
            # In var2
            elif pd.notnull(row['level_var2']) and pd.isnull(row['level_var1']) and pd.isnull(row['var1']):
                return f" when TRIM(CAST({row['var2']} as varchar(999))) = '{row['level_var2']}' then {row['estimate']}"
            ## One numerical variable
            # In var1
            elif pd.notnull(row['var1']) and pd.isnull(row['level_var1']) and pd.isnull(row['var2']) and pd.isnull(row['level_var2']):
                return f" + {row['var1']} * {row['estimate']}"
            # In var2
            elif pd.notnull(row['var2']) and pd.isnull(row['level_var2']) and pd.isnull(row['var1']) and pd.isnull(row['level_var1']):
                return f" + {row['var2']} * {row['estimate']}"
            ## Interaction
            # 2 categorical variables
            elif pd.notnull(row['level_var1']) and pd.notnull(row['level_var2']):
                return f" when TRIM(CAST({row['var1']} as varchar(999))) = '{row['level_var1']}' and TRIM(CAST({row['var2']} as varchar(999))) = '{row['level_var2']}' then {row['estimate']}"
            # 2 numerical variables
            elif pd.notnull(row['var1']) and pd.isnull(row['level_var1']) and pd.notnull(row['var2']) and pd.isnull(row['level_var2']):
                return f" + {row['var1']} * {row['var2']} * {row['estimate']}"
            # 1 categorical and 1 numerical variable, var1 is categorical
            elif pd.notnull(row['var1']) and pd.notnull(row['level_var1']) and pd.notnull(row['var2']) and pd.isnull(row['level_var2']):
                return f" when TRIM(CAST({row['var1']} as varchar(999))) = '{row['level_var1']}' then {row['var2']} * {row['estimate']}"
            # 1 categorical and 1 numerical variable, var2 is categorical
            elif pd.notnull(row['var2']) and pd.notnull(row['level_var2']) and pd.notnull(row['var1']) and pd.isnull(row['level_var1']):
                return f" when TRIM(CAST({row['var2']} as varchar(999))) = '{row['level_var2']}' then {row['var1']} * {row['estimate']}"
            else:
                return '!!!!!! DEFECT !!!!!!'

        # Get the base sql statement for each row
        scoring_table["sql_body"] = scoring_table.apply(get_row_sql_statement, axis=1)

        # Create SQL statement for each row
        for i, row in scoring_table.iterrows():
            sql_body = row["sql_body"]
            # Add to SQL beginning
            # If the row is not the first one and the previous row is not the same as the current one and the current row is a categorical variable
            # Add "case" to the beginning of the SQL statement
            if (i != 0) & (pd.notnull(row["level_var1"]) | pd.notnull(row["level_var2"])) & \
                ((scoring_table.iloc[i-1,0] != row.iloc[0]) | (scoring_table.iloc[i-1,1] != row.iloc[1])):
                sql_body = f" + case {sql_body}"
            # If the row is the first one and current row is a categorical variable
            # Add "case" to the beginning of the SQL statement
            elif (i == 0) & (pd.notnull(row["level_var1"]) | pd.notnull(row["level_var2"])):
                sql_body = f" + case {sql_body}"
            # If the row is not a categorical variable
            # Do not add "case" to the beginning of the SQL statement
            else:
                sql_body = f"{sql_body}"

            # Add to SQL end
            # If the row is not the last one and the next row is not the same as the current one and the current row is a categorical variable
            # Add "else 0.0 end" to the end of the SQL statement
            if (i != len(scoring_table) - 1):
                if ((scoring_table.iloc[i+1,0] != row.iloc[0]) | (scoring_table.iloc[i+1,1] != row.iloc[1])) & \
                    (pd.notnull(row["level_var1"]) | pd.notnull(row["level_var2"])):
                    sql_body = f"{sql_body} else 0.0 end"
            # If the row is the last one and the current row is a categorical variable
            elif (i == len(scoring_table) - 1) & (pd.notnull(row["level_var1"]) | pd.notnull(row["level_var2"])):
                sql_body = f"{sql_body} else 0.0 end"
            # If the row is not a categorical variable
            else:
                sql_body = f"{sql_body}"
            
            # Update the sql_body column
            scoring_table.loc[i,"sql_body"] = sql_body

        # Print SQL statements for each row
        # for i, row in scoring_table.iterrows():
        #     print(row["sql_body"])

        # Create SQL full statement
        sql_full = ""
        for i, row in scoring_table.iterrows():
            sql_full += row["sql_body"] + ("\n" if format else "")

        return sql_full


    def predict_linear(self,X: pd.DataFrame) -> pd.Series:
        """ Predict linear part of the GLM model based on the scoring table and model matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Model matrix as created by the prepare_model_matrix() function.

        Returns
        -------
        pd.Series
            Linear part of the GLM model

        Uses
        ----
        scoring_table : pd.DataFrame
            Required columns: var1, var2, level_var1, level_var2

        model_matrix_columns : list[str]
            List of model matrix column names as created by the _reconstruct_model_matrix_columns() function.
        """
        # Filter X to contain only the columns in the scoring table
        # If the model matrix does not contain all the columns in the scoring table, raise a warning
        try:
            X_filtered = X[self.model_matrix_columns]
            return np.dot(X_filtered, self.scoring_table["estimate"])
        except KeyError as e:
            warnings.warn(f"""
                Model matrix does not contain all the columns in the scoring table.
                Continuing with the intersection of the columns.
                Original error: {e}
            """)
            # Which columns are missing in the model matrix?
            scoring_table_columns = set(self.model_matrix_columns)
            model_matrix_columns = set(X.columns)
            missing_columns = scoring_table_columns.difference(model_matrix_columns)
            filtr = ~pd.Series(self.model_matrix_columns).isin(missing_columns)

            # Filter X to contain only the columns in the scoring table & filter scoring table
            model_matrix_columns_filtered = pd.Series(self.model_matrix_columns)[filtr].tolist()
            X_filtered = X[model_matrix_columns_filtered]

            print(f"""There is an estimate for the following columns in the scoring table, but they are missing in the model matrix: {missing_columns}
            Please check if the model matrix is correct.""")
            return np.dot(X_filtered, self.scoring_table.loc[filtr,"estimate"])


    def to_sql(self, *args, **kwargs) -> None:
        """ Upload scoring table to the database

        Parameters
        ----------
        **kwargs
            Arguments passed to the pandas.DataFrame.to_sql() function.

        Uses
        ----
        self.scoring_table : pd.DataFrame
            Required columns: var1, var2, level_var1, level_var2, estimate
        """
        # Convert to string where not None
        scoring_table = self.scoring_table
        for column in ['var1', 'var2', 'level_var1', 'level_var2']:
            scoring_table[column] = scoring_table[column].map(lambda x: str(x) if x is not np.nan else None)
        scoring_table.to_sql(*args, **kwargs)

    def to_csv(self, *args, **kwargs) -> None:
        """ Save scoring table as csv file

        Parameters
        ----------
        **kwargs
            Arguments passed to the pandas.DataFrame.to_csv() function.

        Uses
        ----
        self.scoring_table : pd.DataFrame
            Required columns: var1, var2, level_var1, level_var2, estimate
        """
        self.scoring_table.to_csv(*args, **kwargs)
