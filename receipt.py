import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Loading the data
data = pd.read_csv('data_daily.csv')

# Convert the 'Date' column to a pandas datetime object
data['Date'] = pd.to_datetime(data['# Date'])

# Extract year and month from the 'Date' column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Group the data by year and month, and calculate the average Receipt_Count for each month
monthly_avg = data.groupby(['Year', 'Month'])['Receipt_Count'].sum().reset_index()

# Filter the data for the year 2021
data_2021 = monthly_avg[(monthly_avg['Year'] == 2021)]

# Extract features (months) and labels (receipt counts)
X = data_2021['Month'].values.reshape(-1, 1)
y = data_2021['Receipt_Count'].values

# Normalizing the input features
mean_X = np.mean(X)
std_X = np.std(X)
X = (X - mean_X) / std_X

# Normalizing the target values
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y

# Creating an LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compiling the model with Mean Squared Error (MSE) as the loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model with validation split
model.fit(X, y, epochs=200, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping])

# Predicting receipt counts for each month in 2022
months_2022 = np.array(list(range(1, 13))).reshape(-1, 1)
months_2022 = (months_2022 - mean_X) / std_X  # Normalizing the input
predictions_2022 = model.predict(months_2022)
predictions_2022 = (predictions_2022 * std_y) + mean_y  # Denormalizing the output

# Displaying the predictions for 2022
for month, prediction in zip(range(1, 13), predictions_2022):
    print(f"Month: {month}, Predicted Receipt Count: {prediction[0]:.2f}")
model.save("receipt_count.keras")


np.save('mean_X.npy', mean_X)
np.save('std_X.npy', std_X)
np.save('mean_y.npy', mean_y)
np.save('std_y.npy', std_y)
