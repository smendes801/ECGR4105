import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from features import prepare_features, analyze_feature_importance, create_feature_subsets
from models import train_and_evaluate_model, compare_models
from visualization import (
    ensure_images_dir, plot_feature_correlation_matrix, plot_confusion_matrix,
    plot_feature_importance, plot_orbital_patterns, plot_model_comparison_confusion_matrices,
    plot_model_performance_bar
)
from utils import print_analysis_header, print_section, format_metrics, format_feature_list, format_orbital_stats


def main():
    # Ensure images directory exists
    ensure_images_dir()

    # Print header
    print_analysis_header()

    # Data Loading
    print_section("1. Data Processing")
    print("Loading data from sources...")
    df = load_data()
    X, y, features = prepare_features(df)
    print(f"• Total objects processed: {len(df)}")
    print(f"• Features extracted: {len(features)}")

    # Feature Analysis
    print_section("2. Feature Selection")
    correlations = analyze_feature_importance(X, y, features)
    # Plot correlation matrix
    data_with_target = X.copy()
    data_with_target['target'] = y
    plot_feature_correlation_matrix(data_with_target)
    X_selected, selected_features = create_feature_subsets(X, correlations)
    format_feature_list(selected_features)

    # Model Training
    print_section("3. Model Training")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # Model Evaluation
    print_section("4. Model Evaluation")
    metrics = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
    format_metrics(metrics)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_feature_importance(metrics['feature_importance'])

    # Orbital Analysis
    print_section("5. Orbital Analysis")
    format_orbital_stats(X, y)

    # Visualization
    print_section("6. Visualizations")
    print("Generating plots...")
    plot_orbital_patterns(X, y)

    # Model Comparison
    print_section("7. Model Comparison")
    results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
    plot_model_comparison_confusion_matrices(results, y_test)
    plot_model_performance_bar(results)

    print("\n" + "="*50)
    print("Analysis Complete")
    print("="*50 + "\n")

if __name__ == "__main__":
    # main() 
    print("Hello, World!")