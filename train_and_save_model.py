import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Read the dataset
df = pd.read_csv('./SolarPrediction.csv')

# Convert columns to datetime
df['Date'] = pd.to_datetime(df['Data'])
df.drop('Data', axis=1, inplace=True)
df['Time'] = pd.to_datetime(df['UNIXTime'], unit='s')
df['TimeSunRise2'] = pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S')
df['TimeSunSet2'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S')

# Feature engineering: extract useful time-based features
df['Hour'] = df['Time'].dt.hour
df['Minute'] = df['Time'].dt.minute
df['SunRise_Hour'] = df['TimeSunRise2'].dt.hour
df['SunRise_Minute'] = df['TimeSunRise2'].dt.minute
df['SunSet_Hour'] = df['TimeSunSet2'].dt.hour
df['SunSet_Minute'] = df['TimeSunSet2'].dt.minute

# Drop unused columns
df.drop(['Time', 'TimeSunRise', 'TimeSunSet', 'TimeSunRise2', 'TimeSunSet2', 'UNIXTime', 'Date'], axis=1, inplace=True)

# Prepare features and target
X = df.drop('Radiation', axis=1)
y = df['Radiation']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib")
