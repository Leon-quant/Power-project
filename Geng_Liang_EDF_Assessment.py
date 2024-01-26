# -*- coding: UTF-8 -*-
# @Time : 1/4/24 11:57 AM
# @File : EDF_Trading_Project.py
# @Name : LiangGeng
# @Software: PyCharm

import pandas as pd
import numpy as np
import requests
import logging
import shutil
import gzip
from datetime import *
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error



# Pre-set
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('display.precision', 3)
logging.basicConfig(level=logging.INFO)


# Access Request & data return
def dataRequest(year_list):
    baseUrl = 'https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/'
    df_list = []
    for data_year in year_list:
        urlPath = f'{data_year:d}/725090-14739-{data_year:d}.gz'
        save_path = f'hourly temperature data for Boston in {data_year}.gz'
        csv_path = f'hourly temperature data for Boston in {data_year}.csv'
        # request the url by http GET and receive the response
        resp = requests.get(baseUrl + urlPath)
        if resp.status_code == 200:
            logging.info(f'{data_year} hourly temperature data for Boston - Request URL:{baseUrl}{urlPath}')
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            # Data Decompression Download & Read
            with gzip.open(save_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            column_widths = [4, 3, 3, 3, 6, 6, 7, 6, 6, 6, 6, 7]
            df = pd.read_fwf(csv_path, widths=column_widths, header=None)
            print(df.shape)
            # print(len(df))
            df_list.append(df)
        else:
            logging.error(f'Failed to download data for year {data_year}!\n'
                          'Please check the parameters.')
    # print(df_list)
    return df_list


# Data Aggregation
def dataAggregation(df_list):
    df_temp = pd.concat(df_list, ignore_index=True)
    # df_temp = pd.DataFrame()
    # for i in df_list:
    #     df_temp = pd.concat([df_temp, i], ignore_index=True)
    column_names = [
        'Year', 'Month', 'Day', 'Hour',
        'Air temperature', 'Dew point temperature',
        'Sea level pressure', 'Wind direction',
        'Wind speed', 'Total cloud cover',
        'One-hour accumulated precipitation',
        'Six-hour accumulated precipitation',
    ]
    df_temp.columns = column_names
    # df_temp.to_csv('df_temp.csv', index=False)
    # print(df_temp.head(100))
    # print(df_temp.shape)
    return df_temp

# df transform & analysis
def dataTrans(df):
    df['Air temperature'] = (df['Air temperature'] / 10) * (9/5) + 32
    df['Dew point temperature'] = (df['Dew point temperature'] / 10) * (9/5) + 32
    df['Wind speed'] = df['Wind speed']/10
    # df.to_csv('df.csv', index=False)
    return df

def mainProcess(year_list):
    # For Q1
    df_temp = dataAggregation(dataRequest(year_list))


    # For Q2
    df_fah = dataTrans(df_temp)


    # For Q3-1
    # Check the Air temperature column for missing values or values of 0
    missing_or_zero_temps_count = df_fah[(df_fah['Air temperature'].isnull()) | (df_fah['Air temperature'] == 0)].shape[0]
    print('The numbers of missing value: ',missing_or_zero_temps_count)
    if missing_or_zero_temps_count != 0:
        print(df_fah[(df_fah['Air temperature'] == 0) | (df_fah['Air temperature'].isnull())])
    else:
        print('There are no data missing values in the Air temperature column. I will check for outliers (data logging errors).')
    df_fah['Air temperature'].plot(kind='box',figsize=(7,4))
    plt.show()  # Intuitively check for outliers
    mean_temp = df_fah['Air temperature'].mean()
    std_temp = df_fah['Air temperature'].std()
    lower_bound = mean_temp - (4 * std_temp)
    upper_bound = mean_temp + (4 * std_temp)
    outliers_rows = df_fah[(df_fah['Air temperature'] < lower_bound) | (df_fah['Air temperature'] > upper_bound)]
    print('The row of wrong air temperature data is:' + '\n', outliers_rows)   # get the row of wrong data
    df_fah.loc[(df_fah['Air temperature'] < lower_bound) | (df_fah['Air temperature'] > upper_bound), 'Air temperature'] = np.nan
    # Use linear interpolation to correct these values
    df_fah['Air temperature'].interpolate(method='linear', inplace=True)
    # Check the results after interpolation
    interpolated_outliers = df_fah.loc[outliers_rows.index, 'Air temperature']
    print('After interpolation, the modified value of missing air temperature is: ' + ' \n ', interpolated_outliers)
    df_fah['Air temperature'].plot(kind='box', figsize=(7, 4))
    plt.show()  # Intuitively check for outliers


    # For Q3-2
    # Filter the dataframe for July 2021
    july_2021_temps = df_fah[(df_fah['Year'] == 2021) & (df_fah['Month'] == 7)]
    # print(july_2021_temps)
    # Calculate the mean of the 'Air temperature' column
    mean_temp_july_2021 = round(july_2021_temps['Air temperature'].mean(), 3)
    print('The mean air temperature for all observations in July 2021 is: ', mean_temp_july_2021)


    # For Q4
    df_fah['UTC_Time'] = pd.to_datetime(df_fah[['Year', 'Month', 'Day', 'Hour']], utc=True)
    df_fah['EST_Time'] = df_fah['UTC_Time'].dt.tz_convert('US/Eastern')
    start_est = pd.Timestamp('2020-07-04 10:00:00').tz_localize('US/Eastern')
    end_est = pd.Timestamp('2020-07-05 10:00:00').tz_localize('US/Eastern')
    filtered_df = df_fah[(df_fah['EST_Time'] >= start_est) & (df_fah['EST_Time'] < end_est)]
    # print(filtered_df)
    boston_gasday_temp= filtered_df['Air temperature'].mean()
    print('The Boston gas day temperature for July 4th, 2020 is: ',boston_gasday_temp,'degrees Fahrenheit')
    df_fah['EST_Time'] = df_fah['EST_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')


    # For Q5
    file_path = './agt.xlsx'
    df_demand = pd.read_excel(file_path)
    # print(df_demand)
    #raw temp data validation
    # print(df_fah)
    new_order = ['EST_Time','Air temperature','Dew point temperature',
                 'Sea level pressure','Wind direction','Wind speed','Total cloud cover',
                 'One-hour accumulated precipitation','Six-hour accumulated precipitation']
    df_fah = df_fah[new_order]
    # print(df_fah)
    # modify -9999 via forward fill.
    # since we will calculate the ave,we don't wanna be influenced by this extreme value
    rows_with_neg9999 = df_fah.isin([-9999]).any(axis=1)
    num_rows_with_neg9999 = rows_with_neg9999.sum()
    print(f"Number of rows with -9999 before replacement: {num_rows_with_neg9999}")
    # shallow copy for the raw df
    df_fah_copy = df_fah.copy()
    df_fah_copy.replace(-9999, pd.NA, inplace=True)
    df_fah_copy.fillna(method='ffill', inplace=True)
    rows_with_neg9999_after = df_fah_copy.isin([-9999]).any(axis=1)
    num_rows_with_neg9999_after = rows_with_neg9999_after.sum()
    print(f"Number of rows with -9999 after replacement: {num_rows_with_neg9999_after}")
    # print(df_fah_copy)
    # Now, we aggregate the data (average) from 10am to 10 am, then join the temperature_df and demand_df
    df_fah_copy['EST_Time'] = pd.to_datetime(df_fah_copy['EST_Time'])
    df_fah_copy['Custom_Day'] = df_fah_copy['EST_Time'].apply(
        lambda dt: dt if dt.time() >= pd.Timestamp('10:00:00').time() else dt - pd.Timedelta(days=1))
    df_fah_copy['Custom_Day'] = df_fah_copy['Custom_Day'].dt.date
    df_daily_avg = df_fah_copy.groupby('Custom_Day').mean()
    # print(df_daily_avg)
    # Join 2 df
    df_demand['Date'] = pd.to_datetime(df_demand['Date']).dt.date
    df_combined = pd.merge(df_daily_avg, df_demand, left_on='Custom_Day', right_on='Date', how='left')
    df_combined.set_index('Date', inplace=True)
    df_combined = df_combined.drop(df_combined.index[0])
    df_combined['sum_agt'] =  df_combined['Residential/Commercial'] + df_combined['Power Plant']
    # print(df_combined)
    df_combined.to_csv('df_combined.csv', index=True) # visual inspection


    # For Q6
    # initial observation
    # Plotting to see the relationship and trend
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    # Air temperature vs Residential/Commercial
    axes[0, 0].scatter(df_combined['Air temperature'], df_combined['Residential/Commercial'], alpha=0.5)
    axes[0, 0].set_title('Air Temperature vs Residential/Commercial')
    # Air temperature vs Power Plant
    axes[0, 1].scatter(df_combined['Air temperature'], df_combined['Power Plant'], alpha=0.5, color='green')
    axes[0, 1].set_title('Air Temperature vs Power Plant')
    # Air temperature vs sum_agt
    axes[1, 0].scatter(df_combined['Air temperature'], df_combined['sum_agt'], alpha=0.5, color='red')
    axes[1, 0].set_title('Air Temperature vs sum_agt')
    # Residential/Commercial over time
    axes[1, 1].plot(df_combined.index, df_combined['Residential/Commercial'], alpha=0.5, color='orange')
    axes[1, 1].set_title('Residential/Commercial over Time')
    # Power Plant over time
    axes[2, 0].plot(df_combined.index, df_combined['Power Plant'], alpha=0.5, color='purple')
    axes[2, 0].set_title('Power Plant over Time')
    # sum_agt over time
    axes[2, 1].plot(df_combined.index, df_combined['sum_agt'], alpha=0.5, color='brown')
    axes[2, 1].set_title('sum_agt over Time')

    plt.tight_layout()
    plt.show()

    # Plotting sum_agt over time with the composition of Residential/Commercial and Power Plant
    plt.figure(figsize=(15, 6))
    plt.plot(df_combined.index, df_combined['sum_agt'], label='Total AGT Demand', color='black', alpha=0.8)
    plt.plot(df_combined.index, df_combined['Residential/Commercial'], label='Residential/Commercial', color='blue',
             alpha=0.5)
    plt.plot(df_combined.index, df_combined['Power Plant'], label='Power Plant', color='red', alpha=0.5)
    plt.title('Total AGT Demand Over Time with Composition')
    plt.legend()
    # calculate the correlation between Residential/Commercial and Power Plant
    corr_coefficient = df_combined['Residential/Commercial'].corr(df_combined['Power Plant'])
    print('The correlation between Residential/Commercial and Power Plant: ',corr_coefficient)

    corr_matrix = df_combined.corr()
    # Plotting the correlation matrix
    plt.figure(figsize=(15, 18))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of df_combined')

    # Limiting the modeling data to the years 2019-2021
    df_combined.index = pd.to_datetime(df_combined.index)
    modeling_data = df_combined.loc['2019-01-01':'2021-12-31']
    testing_data = df_combined.loc['2022-01-01':'2022-12-31']
    # print(modeling_data)

    # ADF test
    adf_result = adfuller(modeling_data['sum_agt'])
    print('ADF-test p-value: ',adf_result[1])

    # seasonality decomp
    decompose_result = seasonal_decompose(modeling_data['sum_agt'], model='additive', period=365)
    decompose_fig = decompose_result.plot()
    decompose_fig.set_size_inches(12, 8)

    # differencing
    data_diff = modeling_data['sum_agt'].diff().dropna()
    data_seasonal_diff = data_diff.diff(365).dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(data_seasonal_diff, label='Seasonally Differenced Data')
    plt.title('Differenced Time Series')
    plt.xlabel('Date')
    plt.ylabel('Differenced Total Natural Gas Demand')
    plt.legend()

    # ADF-test again
    adf_test_seasonal_diff = adfuller(data_seasonal_diff)
    print(f'ADF p-value after seasonal differencing: {adf_test_seasonal_diff[1]}')

    # ACF PACF
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(data_seasonal_diff, lags=40, ax=axes[0])
    plot_pacf(data_seasonal_diff, lags=40, ax=axes[1], method='ywm')

    # construct model / parameter selection

    model = SARIMAX(modeling_data['sum_agt'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 4),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit()
    print(results.summary())


    pred = results.get_prediction(start=pd.to_datetime('2022-01-01'), end=pd.to_datetime('2022-12-31'), dynamic=False)
    pred_ci = pred.conf_int()
    pred_mean = pred.predicted_mean
    testing_data['predicted_sum_agt'] = pred_mean

    # MSE
    mse = mean_squared_error(testing_data['sum_agt'], testing_data['predicted_sum_agt'])
    print(f'The Mean Squared Error of our forecasts is {mse}')

    # forcast value vs  actual value
    plt.figure(figsize=(10, 5))
    plt.plot(testing_data['sum_agt'], label='Real Total Natural Gas Demand')
    plt.plot(testing_data['predicted_sum_agt'], color='red', label='Predicted Total Natural Gas Demand')
    plt.title('Real vs Predicted Total Natural Gas Demand')
    plt.xlabel('Date')
    plt.ylabel('Total Natural Gas Demand')
    plt.legend()
    plt.show()

    # forcast value  vs  Air temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(testing_data['Air temperature'], testing_data['predicted_sum_agt'], alpha=0.5)
    plt.title('Relationship between Predicted Total Natural Gas Demand and Air Temperature')
    plt.xlabel('Air Temperature')
    plt.ylabel('Predicted Total Natural Gas Demand')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    year_list = [2019, 2020, 2021, 2022, 2023]
    mainProcess(year_list)