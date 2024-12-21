import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib  # Updated import
from .visualize import plot_feature_importance, plot_actual_vs_predicted, plot_residuals, save_predictions


def train_random_forest(data_path, output_path, fine_tune=False):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - data_path: Path to the processed dataset (CSV file).
    - output_path: Path to save the model outputs.
    """
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Drop irrelevant columns
    columns_to_drop = [
        "PHONE", "ADDRESSLINE1", "ADDRESSLINE2", "CONTACTLASTNAME",
        "CONTACTFIRSTNAME", "CUSTOMERNAME", "TERRITORY", "POSTALCODE"
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Encode categorical features
    categorical_columns = [
        "CITY", "STATE", "STATUS", "PRODUCTLINE", "COUNTRY", "DEALSIZE", "PRODUCTCODE"
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Define target and features
    target = "SALES_capped"
    feature_columns = [col for col in df_encoded.columns if col != target]

    X = df_encoded[feature_columns]
    y = df_encoded[target]

    # Handle missing values (if any)
    X = X.fillna(0)  # Avoid inplace operations to prevent warnings

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if fine_tune:
        print("Fine-tuning Random Forest...")
        model, best_params = fine_tune_random_forest(X_train, y_train)
        print(f"Best Hyperparameters: {best_params}")

        # Save fine-tuning results
        fine_tuning_results_path = os.path.join(output_path, "random_forest_fine_tuning_results.txt")
        with open(fine_tuning_results_path, "w") as f:
            f.write(f"Best Hyperparameters: {best_params}\n")
        print(f"Fine-tuning results saved to {fine_tuning_results_path}")
    else:
        print("Training Random Forest with default parameters...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Model Performance:")
    print(f"- Mean Squared Error (MSE): {mse:.2f}")
    print(f"- R-squared (R2): {r2:.2f}")

    model_path = os.path.join(output_path, "random_forest_model.pkl")
    joblib.dump(model, model_path)  # Save the trained model
    print(f"Random Forest model saved to {model_path}")

    # Save evaluation results
    results_path = os.path.join(output_path, "random_forest_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"R-squared (R2): {r2:.2f}\n")

    print(f"Results saved to {results_path}")

    # Visualizations
    visualization_path = os.path.join(output_path, "random_forest_visualizations")
    os.makedirs(visualization_path, exist_ok=True)

    plot_feature_importance(model, X.columns, visualization_path)
    plot_actual_vs_predicted(y_test, y_pred, "Random Forest: Actual vs Predicted", visualization_path)
    plot_residuals(y_test, y_pred, "Random Forest: Residuals", visualization_path)

    # Save predictions
    save_predictions(y_test, y_pred, output_path)

def fine_tune_random_forest(X_train, y_train):
    """
    Fine-tune a Random Forest model using RandomizedSearchCV.

    Parameters:
    - X_train: Training feature data.
    - y_train: Training target data.

    Returns:
    - best_model: The fine-tuned Random Forest model.
    - best_params: The best hyperparameters found during tuning.
    """
    # Hyperparameter grid for RandomizedSearchCV
    param_distributions = {
        "n_estimators": randint(50, 200),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }

    # Perform RandomizedSearchCV
    random_forest = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    # Best model and hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    return best_model, best_params



def predict_random_forest(processed_data_path, model_path, output_path):
    """
    Use a trained Random Forest model to generate predictions.

    Parameters:
    - processed_data_path: Path to the processed dataset (CSV file).
    - model_path: Path to the saved Random Forest model.
    - output_path: Path to save the prediction results.
    """
    # Load the processed data
    data = pd.read_csv(processed_data_path)

    # Drop unnecessary columns (ensure it matches the training process)
    columns_to_drop = [
        "PHONE", "ADDRESSLINE1", "ADDRESSLINE2", "CONTACTLASTNAME",
        "CONTACTFIRSTNAME", "CUSTOMERNAME", "TERRITORY", "POSTALCODE",
        "SALES_capped"  # Drop the target column to avoid mismatch
    ]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Encode categorical features
    categorical_columns = [
        "CITY", "STATE", "STATUS", "PRODUCTLINE", "COUNTRY", "DEALSIZE", "PRODUCTCODE"
    ]
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Load the trained model
    model = joblib.load(model_path)

    # Validate feature consistency
    trained_features = model.feature_names_in_  # Features used during model training
    data_encoded = data_encoded.reindex(columns=trained_features, fill_value=0)  # Align columns

    # Predict
    predictions = model.predict(data_encoded)

    # Save predictions
    data['Predicted_SALES_RF'] = predictions
    data.to_csv(output_path, index=False)
    print(f"Random Forest predictions saved to {output_path}")

if __name__ == "__main__":
    # Example paths (adjust as needed)
    data_path = "data/processed/sales_processed.csv"
    output_path = "models/outputs"
    os.makedirs(output_path, exist_ok=True)
    train_random_forest(data_path, output_path)
