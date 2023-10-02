import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load your time series data
data = pd.read_csv('fall2023datathonset.csv')

# Convert the 'Year' column to DateTime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Define date ranges for training, testing, and CV
train_start = '1900-01-01'
train_end = '1990-01-01'

test_start = '1850-01-01'
test_end = '1899-01-01'

cv_start = '1920-01-01'
cv_end = '2022-01-01'
# Split the data into training, testing, and CV datasets
train_data = data[(data.index >= train_start) & (data.index <= train_end)]
test_data = data[(data.index >= test_start) & (data.index <= test_end)]
cv_data = data[(data.index >= cv_start) & (data.index <= cv_end)]

# Initialize dictionaries to store error metrics
mape_dict = {}
mse_dict = {}
mae_dict = {}

# Define a function to calculate error metrics
def calculate_metrics(actual, forecast):
    mape = (1 / len(actual)) * 100 * np.mean(np.abs((actual - forecast) / actual))
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    return mape, mse, mae

# Training dataset
X_train = train_data.index.year.values.reshape(-1, 1)
y_train = train_data['Anomaly'].values
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
model_train = LinearRegression()
model_train.fit(X_train_poly, y_train)
forecast_train = model_train.predict(X_train_poly)

mape_train, mse_train, mae_train = calculate_metrics(y_train, forecast_train)

mape_dict['Training'] = mape_train
mse_dict['Training'] = mse_train
mae_dict['Training'] = mae_train

# Testing dataset
X_test = test_data.index.year.values.reshape(-1, 1)
y_test = test_data['Anomaly'].values
X_test_poly = poly_features.transform(X_test)
forecast_test = model_train.predict(X_test_poly)

mape_test, mse_test, mae_test = calculate_metrics(y_test, forecast_test)

mape_dict['Testing'] = mape_test
mse_dict['Testing'] = mse_test
mae_dict['Testing'] = mae_test

# Extend the X_range for future years
future_years = np.arange(2023, 2030)  # Replace with the desired range of years
X_future = np.concatenate((X_combined, future_years.reshape(-1, 1)))

# Transform the extended X values
X_future_poly = poly_features.transform(X_future)

# Predict values for the future years
forecast_future = model_train.predict(X_future_poly)


# CV dataset
X_cv = cv_data.index.year.values.reshape(-1, 1)
y_cv = cv_data['Anomaly'].values
X_cv_poly = poly_features.transform(X_cv)
forecast_cv = model_train.predict(X_cv_poly)

mape_cv, mse_cv, mae_cv = calculate_metrics(y_cv, forecast_cv)

mape_dict['CV'] = mape_cv
mse_dict['CV'] = mse_cv
mae_dict['CV'] = mae_cv

# Print the error metrics for each dataset
for dataset, mape in mape_dict.items():
    print(f"MAPE ({dataset}): {mape:.2f}%")
    
for dataset, mse in mse_dict.items():
    print(f"MSE ({dataset}): {mse:.6f}")
    
for dataset, mae in mae_dict.items():
    print(f"MAE ({dataset}): {mae:.6f}")

#Scatter Data
plt.scatter(X_train, y_train, label="Training Data", color='blue')
plt.scatter(X_test, y_test, label="Testing Data", color='red')
plt.scatter(X_cv, y_cv, label="CV Data", color='green')

# Create a range of X values for plotting the regression curve
X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)

# Calculate the predicted values for the regression curve
y_range_pred = model_train.predict(X_range_poly)

# Plot the regression curve
plt.plot(X_range, y_range_pred, label="Regression Curve", color='#f5b042')

# Add labels and a legend to the plot
plt.xlabel('Year')
plt.ylabel('Annualized Anomaly (°F)')
plt.legend()

# Show the plot
plt.show()

# Forecasting for combined datasets
X_combined = np.concatenate((X_train, X_test, X_cv))
y_combined = np.concatenate((y_train, y_test, y_cv))
X_combined_poly = poly_features.transform(X_combined)
forecast_combined = model_train.predict(X_combined_poly)

# Calculate error metrics for the combined forecast
mape_combined, mse_combined, mae_combined = calculate_metrics(y_combined, forecast_combined)

mape_dict['Combined'] = mape_combined
mse_dict['Combined'] = mse_combined
mae_dict['Combined'] = mae_combined

# Print the error metrics for the combined dataset
print(f"MAPE (Combined): {mape_combined:.2f}%")
print(f"MSE (Combined): {mse_combined:.6f}")
print(f"MAE (Combined): {mae_combined:.6f}")

# Plot the combined forecast
plt.figure(figsize=(12, 6))
plt.scatter(X_combined, y_combined, label="Combined Data", color='blue')
plt.plot(X_range, y_range_pred, label="Regression Curve", color='#f5b042')
plt.xlabel('Year')
plt.ylabel('Annualized Anomaly (°F)')
plt.legend()
plt.grid(True)
plt.show()

# Save the combined forecasts to a CSV file
combined_forecasts = pd.DataFrame({'Year': X_combined.flatten(), 'Forecast': forecast_combined})
combined_forecasts.to_csv('combined_forecasts2.csv', index=False)



# Print the coefficients of the quadratic regression model
coefficients = model_train.coef_
intercept = model_train.intercept_
print(f"Quadratic Regression Formula: y = {coefficients[2]:.6f}x^2 + {coefficients[1]:.6f}x + {intercept:.6f}")