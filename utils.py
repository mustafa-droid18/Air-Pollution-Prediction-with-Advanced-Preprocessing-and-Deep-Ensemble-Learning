import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math

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

# # Function to plot distribution of multiple variables on a specific date
# def day_distribution(df, variables, date):
#     day_data = df.loc[date]

#     fig, axes = settings(variables, l=5, n_cols=2)

#     # Plot each variable's distribution on the given date
#     for i, variable in enumerate(variables):
#         ax = axes[i]
#         day_data[variable].plot(marker='o', linestyle='-', ax=ax)
#         ax.set_title(f'{variable} Levels on {date}')
#         ax.set_xlabel('Time')
#         ax.set_ylabel(f'{variable} Level')
#         ax.grid(True)

#     # Remove any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()

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
