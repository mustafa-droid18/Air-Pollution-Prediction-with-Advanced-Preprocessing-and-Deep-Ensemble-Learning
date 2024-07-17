# Problem Statement 
Develop a predictive model for PM2.5 forecasting aimed at providing accurate and timely predictions of particulate matter concentrations over a specified time horizon. The project encompasses data collection, preprocessing, and feature engineering, followed by the selection and training of appropriate machine learning or statistical models. Success will be measured by the model's ability to accurately forecast PM2.5 levels, as assessed through metrics such as mean absolute error (MAE), root mean squared error (RMSE) and R2

# Project Overview

This project focuses on forecasting the particulate matter (PM 2.5) levels for the city of Ghaziabad over a 48-hour period using an ensemble learning model.

## Data Collection and Preprocessing

- **Data Source**: Central Pollution Control Board (CPCB) website.
- **Time Period**: January 2017 - December 2021.
- **Data Points**: 33,217 values.
- **Preprocessing**:
  - **Imputation**: Probabilistic Principal Component Analysis (PPCA).
  - **Normalization**: Robust Scaler.

## Tools and Technologies

- **Programming Language**: Python
- **Frameworks and Libraries**:
  - TensorFlow
  - Scikit-Learn
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Bi-LSTM model
  - XGBoost Regressor

## Key Findings

- **Imputation**: PPCA outperformed other methods in filling missing values.
- **Normalization**: Robust Scaler was the most efficient for normalizing time-series data.
- **Model Performance**: The ensemble method achieved an R² score of 0.92, surpassing the individual models.

# Instructions to Run the Code

## Step 1: Data Formatting

Before running the preprocessing file and predictive model, ensure the datasets are formatted correctly. Use the `Data_Formating.ipynb` notebook provided in the repository to format the datasets appropriately. You may need to modify the code to suit your specific data. 

Ensure that the final dataset to be used contains only numerical data. Additionally, there should be a column with datetime values formatted correctly to be used as an index for time-series analysis in future steps.

## Step 2: Data Preprocessing

In this step, we conduct data preprocessing tasks such as imputing missing data and feature engineering. Utilize the `Data_Preprocessing.ipynb` notebook to impute missing values and engineer features that might enhance the predictive model's performance.

## Step 3: Data Visualization Without Imputation

In this step, we focus on understanding the dataset's structure and identifying potential patterns, trends, and outliers through various visualization techniques. This step is crucial as it helps in identifying the underlying distribution and relationships within the data without applying any imputation methods to handle missing values. Utilize the `Data_Visualization_WOI.ipynb` notebook to create and analyze visual representations of the raw data.

## Step 4: Data Visualization

The `Data_Visualization.ipynb` notebook provides a comprehensive exploration of the dataset after preprocessing. This step includes visualizing the trends, seasonal patterns, and potential anomalies in the PM2.5 data. Key visualizations include time series plots, seasonal decomposition, and geographical plots if location data is available. These visualizations help in understanding the data better and guide feature engineering for the predictive models.

## Step 5: Model Training and Prediction

The `Models_Predictions.ipynb` notebook is dedicated to training various machine learning and statistical models for PM2.5 forecasting. This includes selecting appropriate models, tuning hyperparameters, and evaluating model performance using cross-validation techniques. In this project, we focus on advanced models like Bi-directional Long Short-Term Memory (Bi-LSTM) networks and XGBoost. The notebook provides code and explanations for each model, guiding you through the process of building robust predictive models.

## Step 6: Results Analysis and Interpretation

The `Results.ipynb` notebook focuses on analyzing the performance of the trained models. This includes calculating evaluation metrics such as MAE, RMSE, and R² to compare model predictions against actual PM2.5 values. Additionally, the notebook provides visualizations of model performance, such as prediction vs. actual plots and error distributions. These analyses help in interpreting the strengths and weaknesses of each model and selecting the best-performing model for deployment.

# Utility Functions (`utils.py`)

The `utils.py` script contains various utility functions essential for data analysis and preprocessing. These functions include loading and preprocessing CSV files (`load_csv`), setting up subplots for multiple variables (`settings`), plotting correlation matrices (`correlation`), visualizing monthly pollutant levels (`plot_monthly_levels`), and creating various distributions such as monthly, frequency, and day-wise distributions (`monthly_distribution`, `freq_distribution`, `day_distribution`). Additionally, the script provides functions for plotting seasonality (`seasonality_plot`), hourly distributions (`hour_wise_distribution`), monthly distributions (`month_wise_distribution`), weekly distributions (`week_wise_distribution`), and day-of-week distributions (`day_of_week_wise_distribution`). The `scaling` function scales features using specified scalers, and the `train_test_split` function prepares the data for modeling by splitting it into training and testing sets. The `data_formating` function formats the data for model training, ensuring it is in the correct shape for machine learning algorithms. Lastly, the script includes functions for plotting training and validation loss (`plot_loss`), evaluating model results (`results`), and saving predictions (`save_predictions`).
