import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_random_forest(data_path):
    """
    Train and evaluate a Random Forest model on the provided dataset.

    Parameters:
    - data_path: Path to the preprocessed dataset (CSV file).
    """
    # Step 1: Load the dataset
    df = pd.read_csv(data_path)

    # Step 2: Define features and target variable
    target = "SALES_capped"  # Replace with the correct target field
    features = [
        "CITY", "STATE", "ORDERNUMBER", "QUANTITYORDERED_capped",
        "PRICEEACH_capped", "ORDERLINENUMBER", "STATUS", "QTR_ID",
        "MONTH_ID", "YEAR_ID", "PRODUCTLINE", "MSRP_capped", "COUNTRY",
        "DEALSIZE"
    ]

    # Ensure all necessary columns exist
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing columns in dataset: {missing_features}")

    # Encode categorical features (if necessary)
    categorical_columns = ["CITY", "STATE", "STATUS", "PRODUCTLINE", "COUNTRY", "DEALSIZE"]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Extract features and target
    X = df_encoded[features]
    y = df[target]

    # Step 3: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    # joblib.dump(model, "random_forest_model.pkl")
    print("Training complete!")


if __name__ == "__main__":
    # Path to the preprocessed dataset
    data_path = "data/processed/sales_processed.csv"
    train_random_forest(data_path)
