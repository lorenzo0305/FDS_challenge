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
    Effectue une validation croisée stratifiée sur le modèle donné.
    
    Args:
        model : scikit-learn estimator déjà entraîné (ou pipeline)
        X_train : DataFrame ou array des features
        y_train : Series ou array des labels
        n_splits : nombre de plis pour StratifiedKFold
        random_state : graine pour reproductibilité
        
    Returns:
        cv_scores : array des scores Accuracy pour chaque pli
        mean_score : moyenne des scores
        std_score : écart-type des scores
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
