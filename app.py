from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Loading the dataset
data = pd.read_csv("data_daily.csv")

# Preprocessing data
data['Date'] = pd.to_datetime(data['# Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Aggregate data by month and year
monthly_data = data.groupby(['Year', 'Month'])['Receipt_Count'].sum().reset_index()

# Splitting data into 2021 and 2022
train_data = monthly_data[monthly_data['Year'] == 2021]

# Loading the trained LSTM model
model = tf.keras.models.load_model("receipt_count.keras")
mean_X = np.load('mean_X.npy')
std_X = np.load('std_X.npy')
mean_y = np.load('mean_y.npy')
std_y = np.load('std_y.npy')
@app.route('/')
def index():
    return render_template('index.html', months=range(1, 13), selected_month=None, prediction=None, plot_url_2021=None, plot_url_2022=None)


@app.route('/predict', methods=['POST'])
def predict():
    selected_month = int(request.form['month'])
    normalized_month = (np.array([selected_month]) - mean_X) / std_X
    normalized_prediction = model.predict(normalized_month.reshape(-1, 1))[0][0]
    prediction = (normalized_prediction * std_y) + mean_y
    # prediction = int(prediction)
    prediction = int(round(prediction))

    # Generating a plot for visualizing the 2021 data
    plt.figure(figsize=(8, 6))
    plt.plot(train_data['Month'], train_data['Receipt_Count'], label='2021 Data', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.title('Receipt Count for 2021')
    plt.legend()

    # Saving the 2021 plot to a BytesIO object
    plot_buf_2021 = BytesIO()
    plt.savefig(plot_buf_2021, format='png')
    plot_buf_2021.seek(0)
    plot_data_2021 = base64.b64encode(plot_buf_2021.read()).decode('utf-8')
    plt.close()

    # Generating a plot for visualizing the 2022 predictions
    months_2022 = np.array(list(range(1, 13))).reshape(-1, 1)
    normalized_months_2022 = (months_2022 - mean_X) / std_X  # Normalizing the months
    normalized_predictions_2022 = model.predict(normalized_months_2022)
    predictions_2022 = (normalized_predictions_2022 * std_y) + mean_y
    plt.figure(figsize=(8, 6))
    plt.plot(months_2022.flatten(), predictions_2022.flatten(), 'ro-', label='2022 Predictions')  # Used the denormalized predictions here
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.title('Receipt Count Predictions for 2022')
    plt.legend()

    # Save the 2022 plot to a BytesIO object
    plot_buf_2022 = BytesIO()
    plt.savefig(plot_buf_2022, format='png')
    plot_buf_2022.seek(0)
    plot_data_2022 = base64.b64encode(plot_buf_2022.read()).decode('utf-8')
    plt.close()

    return render_template('index.html', months=range(1, 13), selected_month=selected_month, prediction=prediction, plot_url_2021=plot_data_2021, plot_url_2022=plot_data_2022)

if __name__ == '__main__':
    app.run(debug=True)
