# Direct Multi-Step Neural Network (DMSNN)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Time series data
time = np.arange(1, 409)

actual_values = np.array([34,7.1,13.2,0.4,3.5,0,0,0,0,0,5.6,27.7,2.7,40.2,39.8,45.1,8.9,0,0,0,0,0.2,13.2,7.9,0,5.9,5.3,0.9,2,0,0,0,0,14.1,2.1,19.6,36.5,18.5,40.9,32.4,0.001,0,0,0,0.1,2.8,1.5,50.2,32.5,10.7,40.9,0.6,0.001,0,0,0,0,0.001,56.7,4.2,17.6,30.9,30.5,0.9,0.001,0,0,0,0,4.6,36.1,3.2,8.4,17.6,10.2,1.1,4.3,0.6,0,0,0,0,25.6,20.4,102.9,6.5,3.4,59.1,2.4,0,0,0,0.001,6.1,0.6,11.5,19.7,10.2,33.5,7.6,0.1,0,0,0,0.9,7.3,41.3,32.3,2.4,48,9.4,15,0.6,0.001,0,0,0,0.001,0.001,21.3,40.2,9.6,22.9,9.1,7,0,0,0,0.001,0.001,1.7,7.5,8.5,8.7,3.2,6.4,0.6,0.001,0,0,0,7.1,44,35.3,42.4,14.1,25.8,1.2,3.2,0,0.001,0,0,0,28.4,0.7,15.7,8.7,1.5,0.8,0.001,0,0.001,0,0,0.001,1,30.8,20.7,0.6,1.2,7.8,0.3,0,0,0,0.001,4.9,2.5,29.6,11.9,17.6,16.4,23.5,0.5,0,0,0,0.1,0.001,6.7,5.4,21.4,3.2,6.4,38.4,2.7,0,0,0,0,3.3,6.1,15,20.4,6.4,60.6,10.8,2.2,0,0,0,0,0.001,7.8,0.001,52.7,34.1,0.001,44.6,2.2,0,0,0,0,11.2,2.4,15.1,32.2,18.8,14.9,24,7.3,0,0,0,0,0.001,0,2,23.7,10.3,1.6,0.001,0.001,0,0,0.001,0.001,16.6,5.8,1.1,4.8,1.4,11.4,11.1,0.001,0.001,0,0,2.1,11.6,15.1,10,1.1,28.1,5.5,10.7,12.6,0,0,0,0,0.001,2.5,32,17.8,25.1,12.4,31,0.3,0.001,0,0,0.001,6.1,0.8,2.5,3.9,9.6,1,5.4,0.001,0,0,0,0,10.7,83.2,70.6,70.8,4.9,0.001,0.001,23.4,0,0,0,0,4,172.7,20.9,35.8,6.8,23.6,14.3,0.001,0.001,0,0,0,4.6,19,3.9,8.2,6.9,26.1,0,4.5,0,0,0.001,0,84.9,32.1,28.2,4.3,28.3,26.1,11.7,3.8,0,0,0,0,0.05,0.05,30.3,9.3,11.3,41.8,7.5,0.1,0,0,0,0,0,1.6,0.05,0.9,88.4,2.8,80.5,9,0,0,0,0,15.1,60.4,27.1,49.8,15.5,38.1,11.4,0.7,0,0,0,0,0,0.3,21.2,36.1,6.4,22.9,3.2,0,0,0,0,0,0,84.2,2.8,1.7,17.9,2.8,0.8,0,0,0,0,0,0,1.3,0.5])

# Prepare the Data
def prepare_data(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 50
X, Y = prepare_data(actual_values, look_back)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the DMSNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(look_back,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

# Predict
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time[-len(y_test):], y_test, label='Actual Values')
plt.plot(time[-len(predictions):], predictions, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# R² Score
r2 = r2_score(y_test, predictions)
print("Coefficient of determination (R^2):", r2)
