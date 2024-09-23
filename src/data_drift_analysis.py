import pandas as pd
import joblib
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from data_ingestion import load_bike_sharing_data
import os

# Feature selection (numerical variables only)
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']

def load_model(model_filename: str):
    """
    Load the saved RandomForest model from file.
    """
    return joblib.load(model_filename)

def prepare_last_week_data(data: pd.DataFrame):
    """
    Prepare the last week of data (week 3: '2011-02-15 00:00:00' to '2011-02-21 23:00:00')
    for data drift analysis.
    return  -> pd.DataFrame
    """
    last_week_data = data.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00']
    return last_week_data

def generate_data_drift_report(reference_data, current_data):
    """
    Generate a data drift report for the last week using only numerical features.
    """
    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features

    # Initialize the data drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])

    # Run the report with reference (January) and current (last week) data
    data_drift_report.run(reference_data=reference_data[numerical_features], 
                          current_data=current_data[numerical_features], 
                          column_mapping=column_mapping)

    # Save the report as an HTML file
    report_path = os.path.join('reports', "data_drift_report_last_week.html")
    data_drift_report.save_html(report_path)
    print("Data drift report saved as data_drift_report_last_week.html")

def main():
    # Load the processed bike sharing data
    data = load_bike_sharing_data()

    # Reference data (January) to compare against
    reference_data = data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']

    # Prepare data for the last week (week 3)
    last_week_data = prepare_last_week_data(data)

    # Generate data drift report for the last week using numerical features
    generate_data_drift_report(reference_data, last_week_data)

if __name__ == "__main__":
    main()
