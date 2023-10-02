import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv('fall2023datathonset.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

#Rename Columns for Prophet Standard Format
data.rename(columns={'Year': 'ds', 'Anomaly': 'y'}, inplace=True)

#Date Ranges
train_start = '1900-01-01'
train_end = '1990-01-01'

test_start = '1850-01-01'
test_end = '1899-01-01'

cv_start = '1920-01-01'
cv_end = '2022-01-01'


#Train, Test, CV
train_data = data[(data['ds'] >= train_start) & (data['ds'] <= train_end)]
test_data = data[(data['ds'] >= test_start) & (data['ds'] <= test_end)]
cv_data = data[(data['ds'] >= cv_start) & (data['ds'] <= cv_end)]

#Metric dictionaries for storage (arrays are slower)
mape_dict = {}
mse_dict = {}
mae_dict = {}

#Metric Func
def calculate_metrics(actual, forecast):
    mape = (1 / len(actual)) * 100 * np.mean(np.abs((actual - forecast) / actual))
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    return mape, mse, mae

#Training 
model_train = Prophet()
model_train.fit(train_data)
future_train = model_train.make_future_dataframe(periods=0)
forecast_train = model_train.predict(future_train)
mape_train, mse_train, mae_train = calculate_metrics(train_data['y'], forecast_train['yhat'])

mape_dict['Training'] = mape_train
mse_dict['Training'] = mse_train
mae_dict['Training'] = mae_train

#Testing 
model_test = Prophet()
model_test.fit(test_data)
future_test = model_test.make_future_dataframe(periods=0)
forecast_test = model_test.predict(future_test)
mape_test, mse_test, mae_test = calculate_metrics(test_data['y'], forecast_test['yhat'])

mape_dict['Testing'] = mape_test
mse_dict['Testing'] = mse_test
mae_dict['Testing'] = mae_test

#CV 
model_cv = Prophet()
model_cv.fit(cv_data)
future_cv = model_cv.make_future_dataframe(periods=0)
forecast_cv = model_cv.predict(future_cv)
mape_cv, mse_cv, mae_cv = calculate_metrics(cv_data['y'], forecast_cv['yhat'])

mape_dict['CV'] = mape_cv
mse_dict['CV'] = mse_cv
mae_dict['CV'] = mae_cv

#Print error metrics 
for dataset, mape in mape_dict.items():
    print(f"MAPE ({dataset}): {mape:.2f}%")
    
for dataset, mse in mse_dict.items():
    print(f"MSE ({dataset}): {mse:.6f}")
    
for dataset, mae in mae_dict.items():
    print(f"MAE ({dataset}): {mae:.6f}")



forecast_cv_df = forecast_cv[['ds', 'yhat']]
forecast_train_df = forecast_train[['ds', 'yhat']]
forecast_test_df = forecast_test[['ds', 'yhat']]


#Plot Raw Data
plt.figure(figsize=(12, 6))
plt.plot(data['ds'], data['y'], label='Actual Data', color='blue')

#Plot Forecasts
plt.plot(forecast_train['ds'], forecast_train['yhat'], label='Training Forecast', color='green')
plt.plot(forecast_test['ds'], forecast_test['yhat'], label='Testing Forecast', color='orange')
plt.plot(forecast_cv['ds'], forecast_cv['yhat'], label='CV Forecast', color='red')

plt.xlabel('Year')
plt.ylabel('Annualized Anomaly (°F)')
plt.title('Prophet Model - Actual vs. Forecast')
plt.legend()
plt.grid(True)
plt.show()


