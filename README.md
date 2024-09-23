# **Bike Sharing Model Monitoring**

## **Project Description**

This project involves monitoring the performance of a RandomForest regression model predicting bike rentals using the UCI Bike Sharing dataset. The workflow employs **Prefect** for orchestrating tasks and **Evidently** for generating reports to monitor data and model drift. The goal is to analyze changes in model performance and detect data drift over time.

The data is divided into different time periods, and we monitor the drift between these periods using Evidently reports. Additionally, we investigate the root causes of drift and propose mitigation strategies.

The purpose of this project is to use evidently to monitor the drift in bike sharing data. The dataset covers 2 complete years (from Jan 2011 and Dec 2012) and we will only work on January and February months.

---

├── src
│ ├── data_drift_analysis.py
│ ├── data_ingestion.py
│ ├── drift_analysis.py
│ ├── model_training.py
│ ├── model_validation.py
│ └── target_drift_analysis.py
│
├── models/ # Directory for saving the trained machine learning models
│ └── random_forest_model.pkl # Trained RandomForest model
│
├── reports # Folder to save Evidently reports
│ ├── model_validation_report.html
│ ├── data_drift_report_last_week.html
│ ├── production_model_drift_report.html
│ ├── target_drift_report.html
│ ├── week_1_drift_report.html # Drift report for week 1
│ ├── week_2_drift_report.html # Drift report for week 2
│ └── week_3_drift_report.html # Drift report for week 3
├── docker-compose.yml # Docker Compose configuration file for running the project
├── main.py # Main script to run the entire workflow
├── Dockerfile # Dockerfile for building the project environment
├── requirements.txt # Python package dependencies
├── README.md # Project documentation (this file)
└── .gitignore # Files to be ignored by Git

---

## **Installation Instructions**

To set up and run the project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone <repo-link>
   cd <repo-directory>
   ```

2. **Docker Setup**:  
   Ensure you have Docker and Docker Compose installed.

3. **Build and Run the Docker Containers**:

   ```bash
   docker-compose up --build
   ```

4. **Python Setup** (Optional if running outside Docker):

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Script**:
   You can run the entire pipeline using:
   ```bash
   python main.py
   ```

## **Usage**

The `main.py` script orchestrates the following tasks:

- Trains a **RandomForest** model on January data.
- Generates drift reports for weeks 1, 2, and 3 in February compared to January.
- Produces detailed reports for model validation, production model drift, and weekly data drift.

## **Methodology**

1. **Step 1: Model Training**  
   A **RandomForestRegressor** is trained on the reference data (January), with features like weather, temperature, humidity, etc. The model is saved in a `models/` directory using **joblib**.

2. **Step 2: Drift Monitoring**  
   Evidently is used to monitor drift between the training data (January) and the subsequent weeks (February). Drift is calculated across the target variable and other features.

3. **Step 3: Weekly Drift Reports**  
   Reports are generated weekly for three weeks in February and saved in a `reports/` directory.

4. **Step 4: Insights from Drift Analysis**  
   After each week, Evidently detects drift, and the results are compared to identify changes across the three weeks.

## **Results**

### **1. Changes Over Weeks 1, 2, and 3**

- **Week 1:**  
  Minimal drift is observed in the numerical features and target distribution. The model is performing similarly to its training period (January).

- **Week 2:**  
  There is noticeable drift in some features, particularly temperature and humidity. The target drift is still under control, but we see the model's predictions slightly deviating from actual values.

- **Week 3:**  
  Significant drift is detected in temperature and humidity, which strongly influences the target variable. The drift report shows a large deviation in the number of bike rentals predicted versus actuals, indicating a drop in model performance.

### **2. Root Cause of Drift**

Based on the data analysis, the root cause of the drift seems to be the changing weather conditions (especially temperature and humidity) in February compared to January. These environmental changes significantly affect the target variable (bike rentals), leading to model prediction errors.

### **3. Mitigation Strategy**

To address the drift, we propose the following strategies:

- **Retraining the Model:**  
  Regularly retrain the model on the most recent data to account for changing weather conditions.

- **Feature Engineering:**  
  Add additional weather-related features or introduce interactions between weather features to better capture their effect on bike rentals.

- **Deploying an Adaptive Learning System:**  
  Implement an online learning approach where the model updates with incoming data in real-time or periodic intervals.

## **Command to Execute**

To execute the entire pipeline, run the following single command:

```bash
docker-compose up --build
```

This command will automatically train the model, generate drift reports, and save them in the appropriate directories.

## **Additional Information**

- **Reports Location**:  
  The drift and validation reports are saved in the `reports/` directory:

  - Model validation report: `model_validation_report.html`
  - Production model drift report: `production_model_drift_report.html`
  - Weekly drift reports: `week_1_drift_report.html`, `week_2_drift_report.html`, `week_3_drift_report.html`

- **Model Location**:  
  The trained RandomForest model is saved in the `models/` directory as `random_forest_model.pkl`.

- **Data Drift Explanation**:  
  Evidently provides a comprehensive analysis of data drift using statistical methods like the Kolmogorov-Smirnov test for numerical features and chi-squared tests for categorical features. The results are visualized in HTML reports.

---
