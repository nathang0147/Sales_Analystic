import os
import pandas as pd
from models.random_forest import predict_random_forest
from models.linear_regression import predict_linear_regression

def main():
    # Define paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(project_root, "../data/processed/sales_processed.csv")
    rf_model_path = os.path.join(project_root, "../models/outputs/random_forest_model.pkl")
    lr_model_path = os.path.join(project_root, "../models/outputs/linear_regression_model.pkl")
    rf_output_path = os.path.join(project_root, "../models/outputs/rf_predictions.csv")
    lr_output_path = os.path.join(project_root, "../models/outputs/lr_predictions.csv")
    unified_predictions_path = os.path.join(project_root, "../models/outputs/predictions.csv")

    # Generate predictions
    print("Generating Random Forest predictions...")
    predict_random_forest(processed_data_path, rf_model_path, rf_output_path)

    print("Generating Linear Regression predictions...")
    predict_linear_regression(processed_data_path, lr_model_path, lr_output_path)

    # Combine predictions
    rf_predictions = pd.read_csv(rf_output_path)
    lr_predictions = pd.read_csv(lr_output_path)
    unified_predictions = pd.concat(
        [rf_predictions[['Predicted_SALES_RF']], lr_predictions[['Predicted_SALES_LR']]], axis=1
    )

    # Save unified predictions
    unified_predictions.to_csv(unified_predictions_path, index=False)
    print(f"Unified predictions saved to {unified_predictions_path}")

if __name__ == "__main__":
    main()
