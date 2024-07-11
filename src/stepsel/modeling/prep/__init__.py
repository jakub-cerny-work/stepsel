'''
The :mod:`stepsel.modeling.prep` module includes functions for data preparation.
'''

from .helper import (
    get_interaction_type,
    relevel_categorical_variable,
    parse_model_formula,
    recognize_variable_types
)

from .interaction import (
    interaction_categorical_numerical,
    interaction_categorical_categorical,
    interaction_numerical_numerical
)

from .model_matrix import (
    prepare_model_matrix
)
