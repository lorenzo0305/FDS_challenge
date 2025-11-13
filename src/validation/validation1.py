from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np


def cross_validate_model(model, X_train, y_train, n_splits=5, random_state=42):
    """
    Performs stratified cross-validation on the given model.

    Args:
        model: trained scikit-learn estimator (or pipeline)
        X_train: DataFrame or array of features
        y_train: Series or array of labels
        n_splits: number of folds for StratifiedKFold
        random_state: seed for reproducibility
    Returns:
        cv_scores: array of Accuracy scores for each fold
        mean_score: mean of the scores
        std_score: standard deviation of the scores
    """
    print("\nPerforming {}-Fold Stratified Cross-Validation on the model...".format(n_splits))
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"Accuracy par pli: {np.round(cv_scores, 4)}")
    print(f"Accuracy moyenne: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_scores, cv_scores.mean(), cv_scores.std()
