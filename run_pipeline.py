import os
import pandas as pd
from models import train_random_forest, train_linear_regression
from models.random_forest import predict_random_forest
from models.linear_regression import predict_linear_regression
from visualizations.context_visualizations import (
    plot_time_based_trend,
    plot_category_based_forecast,
    plot_geographic_forecast,
)


def get_project_root():
    """
    Dynamically determine the root directory of the project.
    """
    return os.path.dirname(os.path.abspath(__file__))


def ensure_directory(path):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def main():
    # Step 1: Define paths
    project_root = get_project_root()
    processed_data_path = os.path.join(project_root, "data", "processed", "sales_processed.csv")
    model_output_path = os.path.join(project_root, "models", "outputs")
    rf_model_path = os.path.join(model_output_path, "random_forest_model.pkl")
    lr_model_path = os.path.join(model_output_path, "linear_regression_model.pkl")
    rf_predictions_path = os.path.join(model_output_path, "rf_predictions.csv")
    lr_predictions_path = os.path.join(model_output_path, "lr_predictions.csv")
    unified_predictions_path = os.path.join(model_output_path, "predictions.csv")
    visualization_output_path = os.path.join(project_root, "visualizations", "outputs")

    # Ensure necessary directories exist
    ensure_directory(model_output_path)
    ensure_directory(visualization_output_path)

    # Step 2: Train models
    print("Running Random Forest model...")
    train_random_forest(processed_data_path, model_output_path, fine_tune=True)

    print("Running Linear Regression model...")
    train_linear_regression(processed_data_path, model_output_path, fine_tune=True)

    # Step 3: Generate predictions
    print("Generating Random Forest predictions...")
    predict_random_forest(processed_data_path, rf_model_path, rf_predictions_path)

    print("Generating Linear Regression predictions...")
    predict_linear_regression(processed_data_path, lr_model_path, lr_predictions_path)

    # Combine predictions with original data
    print("Combining predictions with original data...")
    original_data = pd.read_csv(processed_data_path)
    rf_predictions = pd.read_csv(rf_predictions_path)
    lr_predictions = pd.read_csv(lr_predictions_path)

    # Ensure 'SALES_capped' is included in the predictions_with_context DataFrame
    predictions_with_context = original_data.copy()
    predictions_with_context['Predicted_SALES_RF'] = rf_predictions['Predicted_SALES_RF']
    predictions_with_context['Predicted_SALES_LR'] = lr_predictions['Predicted_SALES_LR']

    # Save unified predictions
    unified_predictions = predictions_with_context[
        ['YEAR_ID', 'MONTH_ID', 'PRODUCTLINE', 'COUNTRY', 'SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    ]
    unified_predictions.to_csv(unified_predictions_path, index=False)
    print(f"Unified predictions saved to {unified_predictions_path}")

    # Step 4: Generate visualizations
    print("Generating visualizations...")
    plot_time_based_trend(unified_predictions, visualization_output_path)
    plot_category_based_forecast(unified_predictions, visualization_output_path)
    plot_geographic_forecast(unified_predictions, visualization_output_path)

    print("Pipeline execution complete!")



if __name__ == "__main__":
    main()
