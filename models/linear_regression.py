import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import joblib
from .visualize import plot_actual_vs_predicted, plot_residuals, save_predictions


def train_linear_regression(data_path, output_path, fine_tune=False):
    """
    Train and evaluate a Linear Regression model on the provided dataset.

    Parameters:
    - data_path: Path to the preprocessed dataset (CSV file).
    - output_path: Path to save the model outputs.
    """
    # Step 1: Load the dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Step 2: Define target and features
    target = "SALES_capped"  # Replace with the correct target field if necessary
    features = [
        "CITY", "STATE", "ORDERNUMBER", "QUANTITYORDERED_capped",
        "PRICEEACH_capped", "ORDERLINENUMBER", "STATUS", "QTR_ID",
        "MONTH_ID", "YEAR_ID", "PRODUCTLINE", "MSRP_capped", "COUNTRY",
        "DEALSIZE", "PRODUCTCODE"
    ]

    # Ensure all necessary columns exist
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    if not available_features:
        raise ValueError(f"No valid features found in dataset. Missing columns: {missing_features}")

    if missing_features:
        print(f"Warning: Missing columns will be skipped: {missing_features}")

    # Encode categorical features
    categorical_columns = [
        col for col in ["CITY", "STATE", "STATUS", "PRODUCTLINE", "COUNTRY", "DEALSIZE", "PRODUCTCODE"]
        if col in df.columns
    ]
    if categorical_columns:
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    else:
        df_encoded = df

    # Update available features to include encoded columns
    all_features = df_encoded.columns.tolist()
    final_features = [col for col in all_features if col != target and col.startswith(tuple(available_features))]

    # Extract features and target
    X = df_encoded[final_features]
    y = df_encoded[target]

    # Handle missing values (if any)
    X = X.fillna(0)

    # Step 3: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if fine_tune:
        print("Fine-tuning Linear Regression with Ridge and Lasso...")
        ridge, ridge_params, lasso, lasso_params = fine_tune_linear_regression(X_train, y_train)
        print(f"Best Ridge Hyperparameters: {ridge_params}")
        print(f"Best Lasso Hyperparameters: {lasso_params}")

        # Save fine-tuning results
        fine_tuning_results_path = os.path.join(output_path, "linear_regression_fine_tuning_results.txt")
        with open(fine_tuning_results_path, "w") as f:
            f.write(f"Best Ridge Hyperparameters: {ridge_params}\n")
            f.write(f"Best Lasso Hyperparameters: {lasso_params}\n")
        print(f"Fine-tuning results saved to {fine_tuning_results_path}")

        # Use Ridge or Lasso for final evaluation (you can choose the better one)
        model = ridge  # You can also use `lasso` based on performance
    else:
        print("Training Linear Regression without regularization...")
        model = LinearRegression()
        model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Model Performance:")
    print(f"- Mean Squared Error (MSE): {mse:.2f}")
    print(f"- R-squared (R2): {r2:.2f}")

    model_path = os.path.join(output_path, "linear_regression_model.pkl")
    joblib.dump(model, model_path)  # Save the trained model
    print(f"Random Forest model saved to {model_path}")

    # Save evaluation results
    results_path = os.path.join(output_path, "linear_regression_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"R-squared (R2): {r2:.2f}\n")

    print(f"Results saved to {results_path}")

    # Visualizations
    visualization_path = os.path.join(output_path, "linear_regression_visualizations")
    os.makedirs(visualization_path, exist_ok=True)

    plot_actual_vs_predicted(y_test, y_pred, "Linear Regression: Actual vs Predicted", visualization_path)
    plot_residuals(y_test, y_pred, "Linear Regression: Residuals", visualization_path)

    # Save predictions
    save_predictions(y_test, y_pred, output_path)


def fine_tune_linear_regression(X_train, y_train):
    """
    Fine-tune Ridge and Lasso models using GridSearchCV.

    Parameters:
    - X_train: Training feature data.
    - y_train: Training target data.

    Returns:
    - best_ridge: The best Ridge model.
    - best_lasso: The best Lasso model.
    - best_params_ridge: The best hyperparameters for Ridge.
    - best_params_lasso: The best hyperparameters for Lasso.
    """
    # Hyperparameter grid for Ridge and Lasso
    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

    # Fine-tune Ridge
    ridge = Ridge()
    ridge_search = GridSearchCV(ridge, param_grid, scoring="neg_mean_squared_error", cv=3)
    ridge_search.fit(X_train, y_train)

    # Fine-tune Lasso
    lasso = Lasso()
    lasso_search = GridSearchCV(lasso, param_grid, scoring="neg_mean_squared_error", cv=3)
    lasso_search.fit(X_train, y_train)

    return (
        ridge_search.best_estimator_,
        ridge_search.best_params_,
        lasso_search.best_estimator_,
        lasso_search.best_params_,
    )

def predict_linear_regression(processed_data_path, model_path, output_path):
    """
    Use a trained Linear Regression model to generate predictions.

    Parameters:
    - processed_data_path: Path to the processed dataset (CSV file).
    - model_path: Path to the saved Linear Regression model.
    - output_path: Path to save the prediction results.
    """
    # Load the processed data
    data = pd.read_csv(processed_data_path)

    # Drop unnecessary columns (ensure it matches the training process)
    columns_to_drop = [
        "PHONE", "ADDRESSLINE1", "ADDRESSLINE2", "CONTACTLASTNAME",
        "CONTACTFIRSTNAME", "CUSTOMERNAME", "TERRITORY", "POSTALCODE",
        "SALES_capped"  # Drop the target column
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
    data['Predicted_SALES_LR'] = predictions
    data.to_csv(output_path, index=False)
    print(f"Linear Regression predictions saved to {output_path}")
