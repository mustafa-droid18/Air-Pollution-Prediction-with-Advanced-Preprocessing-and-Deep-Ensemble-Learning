import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import math
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

# Function to load a CSV file and preprocess it
def load_csv(filepath):
    # Read the CSV file into a DataFrame, treating 'None' as NA values
    df = pd.read_csv(filepath, na_values='None')
    # Convert 'Datetime' column to datetime type and set it as the index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df

# Function to set up subplots for multiple variables
def settings(variables, l=5, n_cols=2):
    n_vars = len(variables)  # Number of variables to plot
    n_rows = math.ceil(n_vars / n_cols)  # Calculate the number of rows needed

    # Create subplots with specified rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, l * n_rows))

    # Flatten axes array if there are multiple rows
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    return fig, axes

# Function to plot correlation matrix heatmap
def correlation(df):
    corr_matrix = df.corr()  # Compute correlation matrix
    f, ax = plt.subplots(figsize=(16, 16))
    
    # Create heatmap of the correlation matrix
    heatmap = sns.heatmap(corr_matrix,
                          square=True,
                          linewidths=.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': .4, 'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={"size": 12})
    
    # Set the tick labels
    ax.set_yticklabels(corr_matrix.columns, rotation=0)
    ax.set_xticklabels(corr_matrix.columns)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

# Function to plot monthly pollutant levels
def plot_monthly_levels(df, pollutants):
    monthly_data = {}

    # Resample the data at monthly intervals and take the mean for each pollutant
    for pollutant in pollutants:
        if pollutant in df.columns:
            monthly_data[pollutant] = df[pollutant].resample('M').mean()
        else:
            print(f"Warning: Pollutant '{pollutant}' not found in DataFrame columns")

    plt.figure(figsize=(10, 6))

    # Plot each pollutant's monthly data
    for pollutant, data in monthly_data.items():
        data.plot(marker='s', linestyle='-', label=pollutant)

    # Customize plot
    plt.title('Monthly Pollutant Levels')
    plt.xlabel('Month')
    plt.ylabel('Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot monthly levels of a specific value from multiple DataFrames
def plot_m(value, df_list, labels, ax):
    for df, label in zip(df_list, labels):
        monthly_value = df[value].resample('M').mean()  # Resample data monthly
        monthly_value.plot(marker='o', linestyle='-', ax=ax, label=label)  # Plot the monthly data
    ax.set_title(f'Monthly {value} Levels (2017-2021)')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'{value} Level')
    ax.legend()
    ax.grid(True)

# Function to plot monthly distribution of multiple variables from multiple DataFrames
def monthly_distribution(df_list, variables, labels=None):
    # Ensure df_list is a list
    if not isinstance(df_list, list):
        df_list = [df_list]

    # If labels are not provided and there are multiple DataFrames, generate default labels
    if labels is None and len(df_list) > 1:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable
    for i, variable in enumerate(variables):
        plot_m(variable, df_list, labels, ax=axes[i])
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot frequency distribution of a specific value from multiple DataFrames
def plot_freq(value, df_list, labels, ax):
    for df, label in zip(df_list, labels):
        sns.histplot(df[value], bins=30, kde=True, ax=ax, label=label, alpha=0.5)
    ax.set_title(f'Distribution of {value}')
    ax.set_xlabel(value)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

# Function to plot frequency distribution of multiple variables from multiple DataFrames
def freq_distribution(df_list, variables, labels=None):
    # Ensure df_list is a list
    if not isinstance(df_list, list):
        df_list = [df_list]

    # If labels are not provided and there are multiple DataFrames, generate default labels
    if labels is None and len(df_list) > 1:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    sns.set(style="whitegrid")
    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable's frequency distribution
    for i, variable in enumerate(variables):
        plot_freq(variable, df_list, labels, ax=axes[i])
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot distribution of a specific variable from multiple DataFrames on a specific date
def plot_day(value, df_list, labels, date, ax):
    for df, label in zip(df_list, labels):
        day_data = df.loc[date]
        day_data[value].plot(marker='o', linestyle='-', ax=ax, label=label)
    ax.set_title(f'{value} Levels on {date}')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{value} Level')
    ax.legend()
    ax.grid(True)

# Function to plot distribution of multiple variables from multiple DataFrames on a specific date
def day_distribution(df_list, variables, date, labels=None):
    # Ensure df_list is a list
    if not isinstance(df_list, list):
        df_list = [df_list]

    # If labels are not provided and there are multiple DataFrames, generate default labels
    if labels == None and len(df_list) > 1:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable's distribution on the given date
    for i, variable in enumerate(variables):
        plot_day(variable, df_list, labels, date, ax=axes[i])
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Function to plot seasonality of a variable within a date range
def seasonality_plot(df, start_date, end_date, var):
    var_range = df.loc[start_date:end_date, var]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=var_range.index, y=var_range, mode='lines+markers', name=var),
                  row=1, col=1)

    fig.update_layout(title=f'Seasonality of {var}',
                      xaxis_title='Date',
                      yaxis_title=f'{var}',
                      showlegend=True)

    fig.show()

# Function to plot hourly distribution of multiple variables
def hour_wise_distribution(df, variables):
    df['hour'] = df.index.hour

    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable's hourly distribution
    for i, variable in enumerate(variables):
        ax = axes[i]
        hourly_distribution = df.groupby('hour')[variable].mean()
        hourly_distribution.plot(kind='bar', ax=ax)
        ax.set_title(f'{variable} Levels by Hour of the Day')
        ax.set_xlabel('Hour of the Day')
        ax.set_ylabel(f'Average {variable} Level')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot monthly distribution of multiple variables
def month_wise_distribution(df, variables):
    df['month'] = df.index.month

    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable's monthly distribution
    for i, variable in enumerate(variables):
        ax = axes[i]
        monthly_distribution = df.groupby('month')[variable].mean()
        monthly_distribution.plot(kind='bar', ax=ax)
        ax.set_title(f'{variable} Levels by Month')
        ax.set_xlabel('Month')
        ax.set_ylabel(f'Average {variable} Level')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot weekly distribution of multiple variables
def week_wise_distribution(df, variables, n_col=2):
    df['week'] = pd.Index(df.index.isocalendar().week)

    fig, axes = settings(variables, l=5, n_cols=n_col)

    # Plot each variable's weekly distribution
    for i, variable in enumerate(variables):
        ax = axes[i]
        weekly_distribution = df.groupby('week')[variable].mean()
        weekly_distribution.plot(kind='bar', ax=ax)
        ax.set_title(f'{variable} Levels by Week')
        ax.set_xlabel('Week')
        ax.set_ylabel(f'Average {variable} Level')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot day-of-week distribution of multiple variables
def day_of_week_wise_distribution(df, variables):
    df['week'] = df.index.dayofweek + 1

    fig, axes = settings(variables, l=5, n_cols=2)

    # Plot each variable's day-of-week distribution
    for i, variable in enumerate(variables):
        ax = axes[i]
        day_of_week_distribution = df.groupby('week')[variable].mean()
        day_of_week_distribution.plot(kind='bar', ax=ax)
        ax.set_title(f'{variable} Levels by Day of the Week')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel(f'Average {variable} Level')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to scale features in a DataFrame
def scaling(df, features_to_scale=[], scaler=RobustScaler(), s=False, r=False):

    # If no specific features are provided, scale all columns
    if features_to_scale == []:
        features_to_scale = df.columns.tolist()

    # Fit the scaler to the specified features in the DataFrame
    scaler.fit(df[features_to_scale])

    # Transform the specified features using the fitted scaler
    scaled_data = scaler.transform(df[features_to_scale])

    # Convert the scaled data array back to a DataFrame with the same index
    scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale, index=df.index)

    # If s is True, print statistics of the scaled 'PM2.5' column
    if s == True:
        print("Max value of scaled 'PM2.5':", max(scaled_df['PM2.5']))
        print("Min value of scaled 'PM2.5':", min(scaled_df['PM2.5']))
        print(scaled_df['PM2.5'].describe())

    if r==True:
        return scaled_df,scaler
    
    return scaled_df

def train_test_split(df):
    
    # Create new columns for day of week, month of year, hour of day, and year
    df.loc[:, 'dow'] = df.index.day_of_week
    df.loc[:, 'moy'] = df.index.month
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'year'] = df.index.year

    # Split the DataFrame into training and testing sets based on the year and month
    df_train = df.loc[(df['year'] < 2021) | (df['moy'] <= 5)].copy()
    df_test = df.loc[(df['year'] >= 2021) & (df['moy'] >= 5)].copy()

    # Filter df_train to include only the last 14 values
    last_14_values = df_train.tail(14)

    # Append the last 14 values from df_train to df_test
    df_test = pd.concat([last_14_values, df_test], axis=0)

    # Drop the 'year' column from both training and testing sets
    df_train.drop(columns=['year'], inplace=True)
    df_test.drop(columns=['year'], inplace=True)

    return df_train, df_test

def data_formating(df,columns=['PM2.5','PM10','NO2','NH3','SO2','CO','Ozone','Temp','RH','WS','WD','dow','moy','hour']):

    df_values=df[columns].values

    #Empty lists to be populated using formatted training data
    X = []
    Y = []

    n_future = 1   # Number of days we want to look into the future based on the past hours.
    n_past = 48  # Number of past hours we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_values has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).

    for i in range(n_past, len(df_values) - n_future +1):
        X.append(df_values[i - n_past:i, 0:df.shape[1]])
        Y.append(df_values[i + n_future - 1:i + n_future, 0])

    X, Y = np.array(X), np.array(Y)

    print('X shape == {}.'.format(X.shape))
    print('Y shape == {}.'.format(Y.shape))

    return X,Y



def plot_loss(history):

    # Assuming history is your training history object from a Keras model
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    fig = gg.Figure()

    fig.add_trace(gg.Scatter(x=list(epochs), y=loss, mode='lines', name='Training loss'))
    fig.add_trace(gg.Scatter(x=list(epochs), y=val_loss, mode='lines', name='Validation loss'))

    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    fig.show()

def results(df_test,model,scaler,model_name='LSTM',scaler_name='Robust Scaler', imputation_method='PPCA'):

    testX,testY= data_formating(df_test)

    if model_name=="XGB":
        testX = testX.reshape(testX.shape[0], -1)

    # Make predictions on the test data
    predictions = model.predict(testX)

    actual_values= testY

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual_values, predictions)
    print('Mean Squared Error (MSE) of Scaled Values:', mse)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = sqrt(mse)
    print('Root Mean Squared Error (RMSE) of Scaled Values:', rmse)

    if model_name=='XGB':
        predictions=predictions.reshape(predictions.shape[0],1)

    unscaled_predictions= np.repeat(predictions, 11, axis=-1)
    unscaled_actual_values=np.repeat(actual_values, 11, axis=-1)

    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)[:,0]
    unscaled_actual_values= scaler.inverse_transform(unscaled_actual_values)[:,0]

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(unscaled_actual_values, unscaled_predictions)
    print('Mean Squared Error (MSE) of Unscaled Values:', mse)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = sqrt(mse)
    print('Root Mean Squared Error (RMSE) of Unscaled Values:', rmse)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot scaled values
    ax1.plot(actual_values, label='Actual (Scaled)', linestyle='-', color='blue')
    ax1.plot(predictions, label='Predicted (Scaled)', linestyle='-', color='orange')
    ax1.set_ylabel('Value')
    ax1.set_title('Scaled Actual vs Predicted')
    ax1.legend()
    ax1.grid(True)

    # Plot unscaled values
    ax2.plot(unscaled_actual_values, label='Actual (Unscaled)', linestyle='-', color='green')
    ax2.plot(unscaled_predictions, label='Predicted (Unscaled)', linestyle='-', color='orange')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Unscaled Actual vs Predicted')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    save_predictions(model_name,scaler_name,imputation_method,unscaled_actual_values,unscaled_predictions)



def save_predictions(model_name,scaler_name,imputation_method,actual_values,predictions):
    
    if model_name=='XGB':
        data=pd.read_csv(f'./Final Results/{imputation_method}/{scaler_name} Predictions.csv')

        data['XGRegressor_Prediction']=predictions
        
        # Saving DataFrame to a CSV file
        data.to_csv(f'./Final Results/{imputation_method}/{scaler_name} Predictions.csv', index=False)

    else:
        
        directory=f'./Final Results/{imputation_method}/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create DataFrame
        df = pd.DataFrame({'Actual Predictions':actual_values, 'LSTM_Prediction':predictions})

        # Save DataFrame to CSV
        df.to_csv(directory+f'{scaler_name} Predictions.csv', index=False)
