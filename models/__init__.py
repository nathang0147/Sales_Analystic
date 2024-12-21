# Import model training functions for simplified access
from .random_forest import train_random_forest
from .linear_regression import train_linear_regression

# Example shared constant
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
}

# Example shared utility function
def log_model_output(model_name, output_path):
    print(f"{model_name} results saved to {output_path}")
