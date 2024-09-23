import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the processed bike sharing data (assumes data_ingestion.py has been run and saved the data)
from data_ingestion import load_bike_sharing_data

# Feature selection and column setup
target = 'cnt'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

def prepare_data(data: pd.DataFrame):
    """
    Prepares the data by splitting it into January (train) and February (test) datasets.
    return  -> pd.DataFrame
    """
    # Split January and February data
    january_data = data.loc['2011-01-01':'2011-01-28']
    return january_data

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains the RandomForestRegressor model on the given training data.
    Returns the trained model.
    return  -> RandomForestRegressor
    """
    # Initialize the model
    model = RandomForestRegressor(random_state=0, n_estimators=50)
    
    # Train the model
    print("Training the RandomForest model...")
    model.fit(X_train, y_train)
    
    return model

def main():
    # Load the processed data
    data = load_bike_sharing_data()

    # Prepare the January data (train)
    january_data = prepare_data(data)

    # Separate the features and target variable
    X = january_data[numerical_features + categorical_features]
    y = january_data[target]
    
    # Split the data into train and test sets (use 30% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train the model on the training data
    model = train_model(X_train, y_train)

    # Save the trained model to a file for later use
    model_filename = 'bike_sharing_random_forest_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    main()
