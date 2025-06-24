import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Read the dataset
df = pd.read_csv('./SolarPrediction.csv')


# Drop unused columns
df.drop(['Time', 'TimeSunRise', 'TimeSunSet', 'UNIXTime', 'Data'], axis=1, inplace=True)

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
