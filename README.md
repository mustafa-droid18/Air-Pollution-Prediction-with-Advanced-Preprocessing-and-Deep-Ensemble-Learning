# Problem Statement 
Develop a predictive model for PM2.5 forecasting aimed at providing accurate and timely predictions of particulate matter concentrations over a specified time horizon. The project encompasses data collection, preprocessing, and feature engineering, followed by the selection and training of appropriate machine learning or statistical models. Success will be measured by the model's ability to accurately forecast PM2.5 levels, as assessed through metrics such as mean absolute error (MAE), root mean squared error (RMSE) and R2

## Instructions to Run the Code

### Step 1: Data Formating

Before running the preprocessing file and predictive model, ensure that the datasets are formatted correctly. Use the `Data_Formating.ipynb` notebook provided in the repository to format the datasets appropriately. You may need to modify the code to suit your specific data. 

Ensure that the final dataset to be used contains only numerical data. Additionally, there should be a column with datetime values formatted correctly to be used as an index for time-series analysis in future steps.

### Step 2: Data Preprocessing

In this step, we conduct data preprocessing tasks such as imputing missing data and feature engineering. Utilize the `Data_Preprocessing.ipynb` notebook to impute missing values and engineer features that might enhance the predictive model's performance.

### Step 3: Data Visualization Without Imputation

In this step, we focus on understanding the dataset's structure and identifying potential patterns, trends, and outliers through various visualization techniques. This step is crucial as it helps in identifying the underlying distribution and relationships within the data without applying any imputation methods to handle missing values. Utilize the `Data_Visualization_WOI.ipynb` notebook to create and analyze visual representations of the raw data.

## Utility Functions (`utils.py`)
The `utils.py` file contains a collection of utility functions designed to facilitate data loading, preprocessing, and visualization tasks for the PM2.5 forecasting project. Key functionalities include loading and formatting CSV data (`load_csv`), setting up plotting configurations (`settings`), and generating various types of visualizations such as correlation heatmaps (`correlation`), monthly pollutant levels (`plot_monthly_levels`), frequency distributions (`freq_distribution`), and time-specific distributions (e.g., `day_distribution`, `hour_wise_distribution`). These utilities streamline the workflow by providing reusable and customizable components for analyzing and visualizing the dataset, thereby supporting the development of an accurate and robust predictive model.