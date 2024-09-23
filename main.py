import pandas as pd
import joblib
import requests
import zipfile
import io
import datetime
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# Ignore warnings
warnings.filterwarnings('ignore')

# URLs and features
BIKE_SHARING_DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
NUMERICAL_FEATURES = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CATEGORICAL_FEATURES = ['season', 'holiday', 'workingday']
TARGET = 'cnt'


def fetch_and_process_data():
    """
    Fetch and process bike-sharing data from the UCI repository.
    return  -> pd.DataFrame
    """
    content = requests.get(BIKE_SHARING_DATA_URL, verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), parse_dates=['dteday'])
    # Indexing with combined datetime for each record
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)), axis=1)
    return raw_data


def train_model(reference_data: pd.DataFrame):
    """
    Train a RandomForest model on the reference data (January).
    """
    X = reference_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = reference_data[TARGET]
    model = RandomForestRegressor(random_state=0, n_estimators=50)
    model.fit(X, y)

    # Ensure the models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save model in the models directory
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model trained and saved as '{model_path}'.")

    return model


def validate_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Validate the RandomForest model using Evidently.
    """
    X_train['target'] = y_train
    X_test['target'] = y_test

    # Predict
    X_train['prediction'] = model.predict(X_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    X_test['prediction'] = model.predict(X_test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

    column_mapping = ColumnMapping(
        target='target',
        prediction='prediction',
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=X_train, current_data=X_test, column_mapping=column_mapping)
    report_path = os.path.join('reports', 'model_validation_report.html')
    report.save_html(report_path)
    print("Model validation report saved as 'model_validation_report.html'.")


def generate_drift_reports(model, reference_data: pd.DataFrame, current_data: pd.DataFrame, weeks):
    """
    Generate drift reports for production model and specific weeks.
    """
    # Generate predictions for the reference data
    reference_data['prediction'] = model.predict(reference_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

    column_mapping = ColumnMapping(
        target=TARGET,
        prediction='prediction',
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    # Production model drift report
    production_drift_report = Report(metrics=[RegressionPreset()])
    production_drift_report.run(reference_data=None, current_data=reference_data, column_mapping=column_mapping)
    report_path = os.path.join('reports', 'production_model_drift_report.html')
    production_drift_report.save_html(report_path)
    print("Production model drift report saved as 'production_model_drift_report.html'.")

    # Weekly reports
    for week, (start, end) in weeks.items():
        week_data = current_data.loc[start:end]
        week_data['prediction'] = model.predict(week_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

        weekly_report = Report(metrics=[RegressionPreset()])
        weekly_report.run(reference_data=reference_data, current_data=week_data, column_mapping=column_mapping)
        report_path = os.path.join('reports', f"week_{week}_drift_report.html")
        weekly_report.save_html(report_path)
        print(f"Week {week} drift report saved as 'week_{week}_drift_report.html'.")


# def generate_target_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
#     """
#     Generate a target drift report using Evidently for the worst week (week 3).
#     """
#     column_mapping = ColumnMapping(target=TARGET, numerical_features=NUMERICAL_FEATURES)

#     target_drift_report = Report(metrics=[TargetDriftPreset()])
#     target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
#     target_drift_report.save_html("target_drift_report.html")
#     print("Target drift report saved as 'target_drift_report.html'.")

def generate_target_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, regressor):
    # Generate predictions for the reference and current datasets
    ref_prediction = regressor.predict(reference_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    cur_prediction = regressor.predict(current_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

    # Add the prediction column to both datasets
    reference_data['prediction'] = ref_prediction
    current_data['prediction'] = cur_prediction

    # Initialize the target drift report
    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])

    # Specify column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = NUMERICAL_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    # Run the report
    target_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    # Save the target drift report
    report_path = os.path.join('reports', 'target_drift_report.html')
    target_drift_report.save_html(report_path)


def generate_data_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, regressor):
    """
    Generate a data drift report for the last week using only numerical features.
    """
    # Generate predictions for the reference and current datasets
    ref_prediction = regressor.predict(reference_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    cur_prediction = regressor.predict(current_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

    # Add the prediction column to both datasets
    reference_data['prediction'] = ref_prediction
    current_data['prediction'] = cur_prediction
    
    column_mapping = ColumnMapping(numerical_features=NUMERICAL_FEATURES)

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report_path = os.path.join('reports', 'data_drift_report_last_week.html')
    data_drift_report.save_html(report_path)
    print("Data drift report saved as 'data_drift_report_last_week.html'.")


def main():
    # Ingest the data
    data = fetch_and_process_data()

    # Split the data
    reference_jan11 = data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
    current_feb11 = data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

    # Train-test split on January data
    X_train, X_test, y_train, y_test = train_test_split(reference_jan11[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
                                                        reference_jan11[TARGET],
                                                        test_size=0.3,
                                                        random_state=0)

    # Train model
    model = train_model(reference_jan11)

    # Validate model
    validate_model(model, X_train, X_test, y_train, y_test)

    # Define week ranges
    weeks = {
        '1': ('2011-01-29 00:00:00', '2011-02-07 23:00:00'),
        '2': ('2011-02-07 00:00:00', '2011-02-14 23:00:00'),
        '3': ('2011-02-15 00:00:00', '2011-02-21 23:00:00')
    }

    # Generate drift reports for weeks
    generate_drift_reports(model, reference_jan11, current_feb11, weeks)

    # Target drift report for worst week (week 3)
    generate_target_drift_report(reference_jan11, current_feb11.loc[weeks['3'][0]:weeks['3'][1]], model)

    # Data drift report for the last week using numerical features
    generate_data_drift_report(reference_jan11, current_feb11.loc[weeks['3'][0]:weeks['3'][1]], model)


if __name__ == "__main__":
    main()
