import pandas as pd
import joblib
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
import os

# Load the processed bike sharing data (assumes data_ingestion.py has been run)
from data_ingestion import load_bike_sharing_data

# Load the trained model
def load_model(model_filename: str):
    """
    Load the saved RandomForest model from file.
    """
    return joblib.load(model_filename)

# Feature selection and column setup
target = 'cnt'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

def prepare_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepares the data by splitting it into January (train) and February (test) datasets.
    """
    # Split January data for training
    january_data = data.loc['2011-01-01':'2011-01-28']

    # Split data into train and test sets
    X_train = january_data[numerical_features + categorical_features]
    y_train = january_data[target]

    # Return prepared data
    return X_train, y_train

def generate_regression_report(model, X_train, y_train):
    """
    Generate a regression report using Evidently to validate the model's performance.
    """
    # Predict the training data
    X_train['prediction'] = model.predict(X_train)

    # Add target values to the dataframe
    X_train['target'] = y_train

    # Define the column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Initialize the regression report using Evidently's RegressionPreset
    regression_report = Report(metrics=[RegressionPreset()])

    # Run the report using training data as reference (X_train is both reference and current for validation)
    regression_report.run(reference_data=X_train, current_data=X_train, column_mapping=column_mapping)

    # Save the report as HTML
    report_path = os.path.join('reports', "regression_validation_report.html")
    regression_report.save_html(report_path)
    print("Regression validation report saved as regression_validation_report.html")

def main():
    # Load the processed data
    data = load_bike_sharing_data()

    # Prepare the data for validation
    X_train, y_train = prepare_data(data)

    # Load the pre-trained model
    model = load_model('bike_sharing_random_forest_model.pkl')

    # Generate the regression validation report
    generate_regression_report(model, X_train, y_train)

if __name__ == "__main__":
    main()
