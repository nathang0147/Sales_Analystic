import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(data_path):
    """
    Train and evaluate a Linear Regression model on the provided dataset.

    Parameters:
    - data_path: Path to the preprocessed dataset (CSV file).
    """
    # Step 1: Load the dataset
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Step 2: Define features and target variable
    target = "SALES"  # Replace with the correct target field if necessary
    features = [
        "CITY", "STATE", "ORDERNUMBER", "QUANTITYORDERED",
        "PRICEEACH", "ORDERLINENUMBER", "STATUS", "QTR_ID",
        "MONTH_ID", "YEAR_ID", "PRODUCTLINE", "MSRP", "COUNTRY",
        "DEALSIZE"
    ]

    # Ensure all necessary columns exist
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    if not available_features:
        raise ValueError(f"No valid features found in dataset. Missing columns: {missing_features}")

    print(f"Missing columns will be skipped: {missing_features}")

    # Encode categorical features (if necessary)
    categorical_columns = [col for col in ["CITY", "STATE", "STATUS", "PRODUCTLINE", "COUNTRY", "DEALSIZE"] if col in df.columns]
    if categorical_columns:
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    else:
        df_encoded = df

    # Update available features to include encoded columns
    all_features = df_encoded.columns.tolist()
    final_features = [col for col in all_features if col not in [target] and col.startswith(tuple(available_features))]

    # Extract features and target
    X = df_encoded[final_features]
    y = df[target]

    # Step 3: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"- Mean Squared Error (MSE): {mse:.2f}")
    print(f"- R-squared (R2): {r2:.2f}")

    # Optional: Save the trained model
    # Uncomment the following lines if you want to save the model
    # import joblib
    # joblib.dump(model, "linear_regression_model.pkl")
    print("Training complete!")

if __name__ == "__main__":
    # Path to the preprocessed dataset
    data_path = "data/sales_processed.csv"
    train_linear_regression(data_path)
