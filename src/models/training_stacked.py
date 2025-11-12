# train_models.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_stacked_model(train_df, test_df, display_cm=True, random_state=42):
    """
    Entraîne un modèle empilé (stacked) avec LGBM, XGB, RF + Logistic Regression.
    Retourne le modèle entraîné et les features utilisées.
    
    Args:
        train_df : DataFrame avec colonnes features + 'player_won'
        test_df  : DataFrame avec les mêmes features
        display_cm : bool, si True affiche la matrice de confusion
        random_state : int, pour reproductibilité

    Returns:
        stack_model : modèle entraîné (StackingClassifier)
        features : liste des colonnes/features utilisées
    """
    # --- Définir les features et cible ---
    features = [c for c in train_df.columns if c not in ['battle_id', 'player_won']]
    X = train_df[features]
    y = train_df['player_won'].astype(int)
    X_test = test_df[features]

    # --- Train / Validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # --- Base learners ---
    base_learners = [
        ('lgbm', lgb.LGBMClassifier(
            objective='binary', learning_rate=0.03, n_estimators=2000,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            eval_metric='logloss', learning_rate=0.05, max_depth=6,
            n_estimators=1500, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, use_label_encoder=False
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=8, n_jobs=-1, random_state=random_state
        ))
    ]

    # --- Meta model ---
    meta_model = LogisticRegression(
        solver='lbfgs', max_iter=4000, random_state=random_state
    )

    # --- Stacking ---
    stack_model = StackingClassifier(
        estimators=base_learners, final_estimator=meta_model,
        stack_method='predict_proba', passthrough=True, n_jobs=-1
    )

    # --- Entraînement ---
    print("\Training stacked model...")
    stack_model.fit(X_train, y_train)

    # --- Évaluation ---
    y_proba = stack_model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    print(f"Validation Results:")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=3))

    # --- Confusion Matrix ---
    if display_cm:
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Lost", "Won"], yticklabels=["Lost", "Won"])
        plt.title("Stacked Model — Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.show()

    return stack_model, features
