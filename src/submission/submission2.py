import pandas as pd
from IPython.display import display

def create_submission2(model, test_df, features, output_path='submission.csv'):
    """
    Génère les prédictions sur le test set et sauvegarde un fichier CSV pour soumission.

    Args:
        model : scikit-learn estimator entraîné (ou pipeline)
        test_df : DataFrame du test set contenant 'battle_id' et les features
        features : liste des colonnes/features à utiliser
        output_path : chemin/fichier de sortie pour le CSV

    Returns:
        submission_df : DataFrame des prédictions
    """
    print("Generating predictions on the test set...")
    
    X_test = test_df[features]
    test_predictions = model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n'{output_path}' file created successfully!")
    display(submission_df.head())
    
    return submission_df
