import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """
    Generate a bar chart of feature importance for Random Forest, displaying top `n` features.

    Parameters:
    - model: Trained Random Forest model.
    - feature_names: List of feature names.
    - output_path: Path to save the visualization.
    - top_n: Number of top features to display (default is 20).
    """
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[-top_n:]  # Get indices of top_n features

    plt.figure(figsize=(10, 8))
    plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title(f"Random Forest Feature Importance (Top {top_n})")
    plt.tight_layout()

    file_path = os.path.join(output_path, "feature_importance.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Feature importance plot saved to {file_path}")


def plot_actual_vs_predicted(y_test, y_pred, title, output_path):
    """
    Generate a scatter plot comparing actual vs predicted values.

    Parameters:
    - y_test: Actual target values.
    - y_pred: Predicted target values.
    - title: Title of the plot.
    - output_path: Path to save the visualization.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit Line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    file_path = os.path.join(output_path, "predicted_vs_actual.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Predicted vs Actual plot saved to {file_path}")


def plot_residuals(y_test, y_pred, title, output_path):
    """
    Generate a histogram of residuals.

    Parameters:
    - y_test: Actual target values.
    - y_pred: Predicted target values.
    - title: Title of the plot.
    - output_path: Path to save the visualization.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(0, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()

    file_path = os.path.join(output_path, "residuals.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Residuals plot saved to {file_path}")


def save_predictions(y_test, y_pred, output_path):
    """
    Save predictions and actual values to a CSV file.

    Parameters:
    - y_test: Actual target values.
    - y_pred: Predicted target values.
    - output_path: Path to save the CSV file.
    """
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    file_path = os.path.join(output_path, "predictions.csv")
    results_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")
