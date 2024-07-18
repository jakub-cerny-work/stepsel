import pytest

# Test group_over_columns function
from stepsel.datasets import load_soccer_data
from stepsel.tools import group_over_columns

@pytest.fixture
def data():
    return load_soccer_data()

@pytest.mark.parametrize("columns, agg_dict, expected_results_dict", [
    (["team"], {"goals": "sum"}, {"columns": ['variable_1', 'level_1', 'goals'],
                                        "n_rows": 16,
                                        "variable_1_value_counts": {'team': 16},
                                        "first_row_value": 48,
                                        "last_row_value": 38
                                        }),
    (["team"], {"goals_opp": "sum"}, {"columns": ['variable_1', 'level_1', 'goals_opp'],
                                            "n_rows": 16,
                                            "variable_1_value_counts": {'team': 16},
                                            "first_row_value": 43,
                                            "last_row_value": 60
                                            }),
    (["yellow"], {"yellow": "count"}, {"columns": ['variable_1', 'level_1', 'yellow'],
                                            "n_rows": 8,
                                            "variable_1_value_counts": {'yellow': 8},
                                            "first_row_value": 60,
                                            "last_row_value": 2
                                            }),
    (["team","hga"], {"goals": ["sum","mean"]}, {"columns": ['variable_1', 'level_1', 'goals_sum', 'goals_mean'],
                                                 "n_rows": 18,
                                                 "variable_1_value_counts": {'team': 16, 'hga': 2},
                                                 "first_row_value": 1.5,
                                                 "last_row_value": 1.6833976833976834
                                                 }),
    (["team","hga"], {"goals": ["sum","mean"], "yellow": "sum"}, {"columns": ['variable_1', 'level_1', 'goals_sum', 'goals_mean', 'yellow_sum'],
                                                                  "n_rows": 18,
                                                                  "variable_1_value_counts": {'team': 16, 'hga': 2},
                                                                  "first_row_value": 81,
                                                                  "last_row_value": 470
                                                                  }),
    (["red",["team","hga","red"],"yellow"], {"goals": ["sum","mean"], "yellow": "sum"}, {"columns": ['variable_1', 'level_1', 'variable_2', 'level_2', 'variable_3', 'level_3', 'goals_sum', 'goals_mean', 'yellow_sum'],
                                                                                         "n_rows": 66,
                                                                                         "variable_1_value_counts": {'team': 56, 'yellow': 8, 'red': 2},
                                                                                         "first_row_value": 955,
                                                                                         "last_row_value": 14
                                                                                         }),
    (["red",["team","hga","red"],"yellow"], {"goals": "sum", "yellow": "sum"}, {"columns": ['variable_1', 'level_1', 'variable_2', 'level_2', 'variable_3', 'level_3', 'goals', 'yellow'],
                                                                                "n_rows": 66,
                                                                                "variable_1_value_counts": {'team': 56, 'yellow': 8, 'red': 2},
                                                                                "first_row_value": 955,
                                                                                "last_row_value": 14
                                                                                }),
    (["red",["team","hga","red"],"yellow"], {"yellow": "count"}, {"columns": ['variable_1', 'level_1', 'variable_2', 'level_2', 'variable_3', 'level_3', 'yellow'],
                                                                  "n_rows": 66,
                                                                  "variable_1_value_counts": {'team': 56, 'yellow': 8, 'red': 2},
                                                                  "first_row_value": 473,
                                                                  "last_row_value": 2
                                                                  }),
])
def test_group_over_columns(data, columns, agg_dict, expected_results_dict):
    actual_result = group_over_columns(data, columns, agg_dict)
    assert actual_result.shape[0] == expected_results_dict["n_rows"], "Number of rows does not match expected value."
    assert actual_result.columns.tolist() == expected_results_dict["columns"], "Column names do not match expected values."
    assert actual_result["variable_1"].value_counts().to_dict() == expected_results_dict["variable_1_value_counts"], "Variable_1 column values do not match expected values."
    assert actual_result.iloc[0,-1] == expected_results_dict["first_row_value"], "First row value does not match expected value."
    assert actual_result.iloc[-1,-1] == expected_results_dict["last_row_value"], "Last row value does not match expected value."

