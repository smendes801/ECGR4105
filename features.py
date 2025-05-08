import pandas as pd
import numpy as np

# --- Feature Engineering and Selection ---

def prepare_features(df):
    """
    Select relevant orbital features and prepare feature matrix and target vector.
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector (1 for debris, 0 for non-debris)
        features (list): List of feature names
    """
    features = [
        'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
        'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'BSTAR', 'MEAN_MOTION_DOT',
        'MEAN_MOTION_DDOT', 'ALTITUDE', 'SSO_FLAG'
    ]
    df_clean = df[features + ['label']].dropna()
    X = df_clean[features]
    y = (df_clean['label'] == 'debris').astype(int)
    return X, y, features


def analyze_feature_importance(X, y, features):
    """
    Analyze feature importance using correlation with the target.
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        features (list): List of feature names
    Returns:
        pd.Series: Correlation values sorted by importance
    """
    data = X.copy()
    data['target'] = y
    correlations = data.corr()['target'].sort_values(ascending=False)
    return correlations


def create_feature_subsets(X, correlations, n_features=7):
    """
    Select the top n_features based on correlation with the target.
    Args:
        X (pd.DataFrame): Feature matrix
        correlations (pd.Series): Feature correlations
        n_features (int): Number of top features to select
    Returns:
        X_selected (pd.DataFrame): Reduced feature matrix
        top_features (list): List of selected feature names
    """
    top_features = correlations.drop('target').nlargest(n_features).index
    X_selected = X[top_features]
    return X_selected, top_features 