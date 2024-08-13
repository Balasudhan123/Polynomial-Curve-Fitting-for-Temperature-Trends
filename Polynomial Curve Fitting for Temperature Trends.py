import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('city_temperature.csv', low_memory=False)
data = data.dropna(subset=['Year', 'Month', 'AvgTemperature'])
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)
data['AvgTemperature'] = data['AvgTemperature'].astype(float)
india_data = data[data['Country'] == 'India']
india_data = india_data[(india_data['Year'] >= 2015) & (india_data['Year'] <= 2019)]
india_data = india_data.groupby(['Year', 'Month'])['AvgTemperature'].mean().reset_index()
X = india_data[['Year', 'Month']].values
y = india_data['AvgTemperature'].values
degree = 4
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
india_data['PredictedTemperature'] = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(y, india_data['PredictedTemperature']))
r2 = r2_score(y, india_data['PredictedTemperature'])
future_year = 2020
future_months = np.arange(1, 13)
future_X = np.array([[future_year, month] for month in future_months])
future_X_poly = poly.transform(future_X)
future_predictions = model.predict(future_X_poly)
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R-squared (R^2): {r2:.2f}")
print("Example of Predicted vs. Actual Values (2015-2019):")
for i in range(min(5, len(india_data))):
    print(f"Year: {india_data['Year'].iloc[i]}, Month: {india_data['Month'].iloc[i]}, Actual: {y[i]:.2f}, Predicted: {india_data['PredictedTemperature'].iloc[i]:.2f}")
print("\nPredicted Values for the First 5 Months of 2020:")
for month, prediction in zip(future_months[:5], future_predictions[:5]):
    print(f"Month: {month:02d}, Predicted Temperature: {prediction:.2f}")
plt.figure(figsize=(12, 6))
x_labels = india_data['Year'].astype(str) + '-' + india_data['Month'].astype(str).str.zfill(2)
plt.scatter(x_labels, india_data['AvgTemperature'], color='blue', label='India Monthly Avg Temperature')
plt.plot(x_labels, india_data['PredictedTemperature'], color='red', linestyle='--', label='Fitted Polynomial Temperature')
future_labels = [f"{future_year}-{month:02d}" for month in future_months]
plt.plot(future_labels, future_predictions, color='green', linestyle='--', label='Predicted Temperature for 2020')
plt.xlabel('Year-Month')
plt.ylabel('Average Temperature')
plt.title('Temperature Trend for India (2015-2019) and Predictions for 2020')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()
