import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ensure_images_dir():
    """
    Ensure the 'images' directory exists for saving plots.
    """
    if not os.path.exists('images'):
        os.makedirs('images')


def save_and_close(fig, filename):
    """
    Save the current figure to the images directory and close it.
    Args:
        fig: Matplotlib figure object
        filename (str): Name of the file to save
    """
    fig.savefig(os.path.join('images', filename), bbox_inches='tight')
    plt.close(fig)


def plot_feature_correlation_matrix(data, filename='feature_correlation_matrix.png'):
    """
    Plot and save the feature correlation matrix heatmap.
    Args:
        data (pd.DataFrame): Data with features and target
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='RdBu', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    save_and_close(fig, filename)


def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    """
    Plot and save a confusion matrix heatmap.
    Args:
        cm (np.ndarray): Confusion matrix
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Debris', 'Debris'],
                yticklabels=['Non-Debris', 'Debris'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_and_close(fig, filename)


def plot_feature_importance(feature_importance, filename='feature_importance.png'):
    """
    Plot and save feature importance barplot.
    Args:
        feature_importance (pd.DataFrame): Feature importance dataframe
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance in Debris Classification')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    save_and_close(fig, filename)


def plot_orbital_patterns(X, y, filename='orbital_patterns.png'):
    """
    Plot and save orbital pattern scatter plots.
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X['MEAN_MOTION'], X['INCLINATION'], c=y)
    plt.xlabel('Mean Motion (orbits/day)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Orbital Pattern Distribution')
    plt.colorbar(label='Is Debris')
    plt.subplot(1, 2, 2)
    plt.scatter(X['ALTITUDE'], X['INCLINATION'], c=y)
    plt.xlabel('Altitude (km)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Altitude vs Inclination Distribution')
    plt.colorbar(label='Is Debris')
    plt.tight_layout()
    save_and_close(fig, filename)


def plot_model_comparison_confusion_matrices(results, y_test, filename='model_comparison_confusion_matrices.png'):
    """
    Plot and save confusion matrices for all models.
    Args:
        results (dict): Model results with predictions
        y_test (pd.Series): True labels
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(15, 10))
    for idx, (name, metrics) in enumerate(results.items()):
        plt.subplot(3, 3, idx + 1)
        cm = metrics.get('confusion_matrix')
        if cm is None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Debris', 'Debris'],
                   yticklabels=['Non-Debris', 'Debris'])
        plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    save_and_close(fig, filename)


def plot_model_performance_bar(results, filename='model_performance_comparison.png'):
    """
    Plot and save a comparative bar chart of model performance metrics.
    Args:
        results (dict): Model results
        filename (str): Output image filename
    """
    fig = plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.2
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        plt.bar(x + i * width, values, width, label=metric.capitalize())
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 1.5, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    save_and_close(fig, filename) 