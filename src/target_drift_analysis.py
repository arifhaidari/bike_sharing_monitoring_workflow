import pandas as pd
import joblib
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from data_ingestion import load_bike_sharing_data
import os

# Feature selection and column setup
target = 'cnt'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

def load_model(model_filename: str):
    """
    Load the saved RandomForest model from file.
    """
    return joblib.load(model_filename)

def prepare_data(data: pd.DataFrame):
    """
    Prepare the worst week of data (week 3: '2011-02-15 00:00:00' to '2011-02-21 23:00:00')
    for target drift analysis.
    return  -> pd.DataFrame
    """
    worst_week_data = data.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00']
    return worst_week_data

def generate_target_drift_report(model, reference_data, current_data):
    """
    Generate a target drift report for the worst week.
    """
    # Predict the current data (worst week)
    current_data['prediction'] = model.predict(current_data[numerical_features + categorical_features])

    # Add the actual target column (cnt) for drift analysis
    current_data['target'] = current_data[target]

    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Initialize the target drift report
    target_drift_report = Report(metrics=[TargetDriftPreset()])

    # Run the report with reference (January) and current (worst week) data
    target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    # Save the report as an HTML file
    report_path = os.path.join('reports', "target_drift_report_worst_week.html")
    target_drift_report.save_html(report_path)
    print("Target drift report saved as target_drift_report_worst_week.html")

def main():
    # Load the processed bike sharing data
    data = load_bike_sharing_data()

    # Load the pre-trained model
    model = load_model('bike_sharing_random_forest_model.pkl')

    # Reference data (January) to compare against
    reference_data = data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']

    # Prepare data for the worst week
    worst_week_data = prepare_data(data)

    # Generate target drift report for the worst week
    generate_target_drift_report(model, reference_data, worst_week_data)

if __name__ == "__main__":
    main()
