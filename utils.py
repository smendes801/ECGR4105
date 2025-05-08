import os

def ensure_images_dir():
    """
    Ensure the 'images' directory exists for saving plots.
    """
    if not os.path.exists('images'):
        os.makedirs('images')


def print_analysis_header():
    """
    Print the main analysis header.
    """
    print("\n" + "="*50)
    print("SPACE DEBRIS CLASSIFICATION ANALYSIS")
    print("="*50 + "\n")


def print_section(title):
    """
    Print a section header.
    Args:
        title (str): Section title
    """
    print("\n" + "-"*20)
    print(f"{title}")
    print("-"*20)


def format_metrics(metrics):
    """
    Print formatted model performance metrics.
    Args:
        metrics (dict): Model metrics
    """
    print("\nModel Performance:")
    print(f"  • Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  • Precision: {metrics['precision']:.2%}")
    print(f"  • Recall:    {metrics['recall']:.2%}")
    print(f"  • F1 Score:  {metrics['f1']:.2%}")


def format_feature_list(features):
    """
    Print a formatted list of selected features.
    Args:
        features (list): List of feature names
    """
    print("\nSelected Features:")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")


def format_orbital_stats(X, y):
    """
    Print formatted orbital statistics for debris and non-debris objects.
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
    """
    debris = X[y == 1]
    non_debris = X[y == 0]
    print("\nOrbital Statistics:")
    print("\nDebris Objects:")
    print(f"  • Count: {len(debris)}")
    print(f"  • Altitude Range: {debris['ALTITUDE'].min():.1f} - {debris['ALTITUDE'].max():.1f} km")
    print(f"  • Mean Motion: {debris['MEAN_MOTION'].mean():.2f} orbits/day")
    print(f"  • SSO Objects: {debris['SSO_FLAG'].sum()}")
    print("\nNon-Debris Objects:")
    print(f"  • Count: {len(non_debris)}")
    print(f"  • Altitude Range: {non_debris['ALTITUDE'].min():.1f} - {non_debris['ALTITUDE'].max():.1f} km")
    print(f"  • Mean Motion: {non_debris['MEAN_MOTION'].mean():.2f} orbits/day")
    print(f"  • SSO Objects: {non_debris['SSO_FLAG'].sum()}") 