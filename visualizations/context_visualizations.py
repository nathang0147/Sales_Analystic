import matplotlib.pyplot as plt
import pandas as pd


def plot_time_based_trend(predictions_with_context, output_path):
    """
    Plot time-based trends of Actual vs Predicted values.

    Parameters:
    - predictions_with_context: DataFrame with time attributes (e.g., YEAR_ID, MONTH_ID).
    - output_path: Path to save the visualization.
    """
    # Check required columns
    required_columns = ['YEAR_ID', 'MONTH_ID', 'SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    for col in required_columns:
        if col not in predictions_with_context.columns:
            raise ValueError(f"Missing required column: {col}")

    # Aggregate predictions and actuals by time
    time_trends = predictions_with_context.groupby(['YEAR_ID', 'MONTH_ID'])[
        ['SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    ].sum().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(time_trends['MONTH_ID'], time_trends['SALES_capped'], label='Actual Sales', marker='o', alpha=0.7)
    plt.plot(time_trends['MONTH_ID'], time_trends['Predicted_SALES_RF'], label='RF Predicted Sales', marker='x', alpha=0.7)
    plt.plot(time_trends['MONTH_ID'], time_trends['Predicted_SALES_LR'], label='LR Predicted Sales', marker='s', alpha=0.7)
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.title("Monthly Sales Trend: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    file_path = f"{output_path}/time_based_trend.png"
    plt.savefig(file_path)
    plt.close()
    print(f"Time-based trend plot saved to {file_path}")


def plot_category_based_forecast(predictions_with_context, output_path):
    """
    Plot Actual vs Predicted values grouped by categories (e.g., PRODUCTLINE).

    Parameters:
    - predictions_with_context: DataFrame with categorical attributes (e.g., PRODUCTLINE).
    - output_path: Path to save the visualization.
    """
    # Check required columns
    required_columns = ['PRODUCTLINE', 'SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    for col in required_columns:
        if col not in predictions_with_context.columns:
            raise ValueError(f"Missing required column: {col}")

    # Aggregate predictions and actuals by category
    category_trends = predictions_with_context.groupby('PRODUCTLINE')[
        ['SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    ].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.bar(category_trends['PRODUCTLINE'], category_trends['SALES_capped'], alpha=0.7, label='Actual Sales')
    plt.bar(category_trends['PRODUCTLINE'], category_trends['Predicted_SALES_RF'], alpha=0.7, label='RF Predicted Sales')
    plt.bar(category_trends['PRODUCTLINE'], category_trends['Predicted_SALES_LR'], alpha=0.7, label='LR Predicted Sales')
    plt.xlabel("Product Line")
    plt.ylabel("Average Sales")
    plt.title("Category-Based Forecast: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    file_path = f"{output_path}/category_based_forecast.png"
    plt.savefig(file_path)
    plt.close()
    print(f"Category-based forecast plot saved to {file_path}")


def plot_geographic_forecast(predictions_with_context, output_path):
    """
    Plot Actual vs Predicted values grouped by geographic attributes (e.g., COUNTRY).

    Parameters:
    - predictions_with_context: DataFrame with geographic attributes (e.g., COUNTRY).
    - output_path: Path to save the visualization.
    """
    # Check required columns
    required_columns = ['COUNTRY', 'SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    for col in required_columns:
        if col not in predictions_with_context.columns:
            raise ValueError(f"Missing required column: {col}")

    # Aggregate predictions and actuals by geographic attribute
    geographic_trends = predictions_with_context.groupby('COUNTRY')[
        ['SALES_capped', 'Predicted_SALES_RF', 'Predicted_SALES_LR']
    ].sum().reset_index()

    plt.figure(figsize=(12, 6))
    plt.bar(geographic_trends['COUNTRY'], geographic_trends['SALES_capped'], alpha=0.7, label='Actual Sales')
    plt.bar(geographic_trends['COUNTRY'], geographic_trends['Predicted_SALES_RF'], alpha=0.7, label='RF Predicted Sales')
    plt.bar(geographic_trends['COUNTRY'], geographic_trends['Predicted_SALES_LR'], alpha=0.7, label='LR Predicted Sales')
    plt.xlabel("Country")
    plt.ylabel("Total Sales")
    plt.title("Geographic-Based Forecast: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    file_path = f"{output_path}/geographic_forecast.png"
    plt.savefig(file_path)
    plt.close()
    print(f"Geographic-based forecast plot saved to {file_path}")
