import pandas as pd
from IPython.display import display

def create_submission2(model, test_df, features, output_path='submission.csv'):
    """
    Generates predictions on the test set and saves a CSV file for submission.
    Args:
        model: trained scikit-learn estimator (or pipeline)
        test_df: DataFrame of the test set containing 'battle_id' and the features
        features: list of columns/features to use
        output_path: output path/filename for the CSV file
    Returns:
        submission_df: DataFrame containing the predictions
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
