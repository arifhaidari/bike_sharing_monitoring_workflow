import pandas as pd
import joblib
from evidently.metric_preset import RegressionPreset, TargetDriftPreset
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from data_ingestion import load_bike_sharing_data
import os

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

def prepare_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Prepare data for drift analysis by selecting weeks 1, 2, and 3 of February.
    """
    week1_data = data.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00']
    week2_data = data.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00']
    week3_data = data.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00']

    return week1_data, week2_data, week3_data

def generate_drift_report(model, reference_data, current_data, week_name):
    """
    Generate a drift report using Evidently for a given week of February.
    """
    # Predict the current data
    current_data['prediction'] = model.predict(current_data[numerical_features + categorical_features])

    # Add target column (actual data) for drift analysis
    current_data['target'] = current_data[target]

    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Initialize regression drift report
    regression_drift_report = Report(metrics=[RegressionPreset()])

    # Run the report with reference (January) and current (February week) data
    regression_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    # Save the report as an HTML file
    report_path = os.path.join('reports', f"model_drift_report_{week_name}.html")
    regression_drift_report.save_html(report_path)
    print(f"Model drift report saved as {report_filename}")

def generate_target_drift_report(model, reference_data, current_data):
    """
    Generate a target drift report for the worst week.
    """
    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.numerical_features = numerical_features

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

    # Reference data (January data) to compare against
    reference_data = data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']

    # Prepare the current data for February weeks
    week1_data, week2_data, week3_data = prepare_data(data)

    # Generate drift reports for each week
    generate_drift_report(model, reference_data, week1_data, "week1")
    generate_drift_report(model, reference_data, week2_data, "week2")
    generate_drift_report(model, reference_data, week3_data, "week3")

    # Perform target drift analysis on the worst week (week3)
    generate_target_drift_report(model, reference_data, week3_data)

if __name__ == "__main__":
    main()
