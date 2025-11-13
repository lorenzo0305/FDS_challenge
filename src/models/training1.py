from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



def train_logistic_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Trains a logistic regression model with GridSearch on train_df, returns the best model and the features used.
    """
    # Features and target 
    features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
    X_train = train_df[features]
    y_train = train_df['player_won']
    X_test = test_df[features]

    # Pipeline : scaling and logistic regression 
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=42,
            solver='saga',
            penalty='elasticnet',
            max_iter=3000
        )
    )

    # Hyperparameters choices
    param_grid = {
        'logisticregression__C': [0.1, 0.5, 1, 2, 5],
        'logisticregression__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    # GridSearchCV 
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("Running Grid Search (Logistic Regression)...")
    grid_search.fit(X_train, y_train)

    print("\n Grid Search complete.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # Training with the best paramameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    print("\nModel retrained with best hyperparameters.")

    # Evaluate on the train
    train_acc = best_model.score(X_train, y_train)
    print(f"Training Accuracy (with best params): {train_acc:.4f}")

    return best_model, features
