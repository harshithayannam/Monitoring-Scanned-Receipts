# Monitoring-Scanned-Receipts
Goal/Purpose:
To forecast the monthly volume of receipts to be scanned in 2022 by examining the trends observed in a dataset containing daily receipt counts from 2021.

Prerequisites:
Python 3.x(latest version if available)
Internet connection for downloading Python and necessary libraries.

Installation and Setup:
Install Python: Download and install Python from python.org. Ensure to add Python to PATH during installation.
Install Libraries: Open a command line interface and simply load the requirements.txt” by running this command in your terminal:
”pip install -r requirements. txt”
Prepare Project Files: Place the provided “receipts.py”, “app.py”, and “data_daily.csv” in a single folder. Inside this folder, create a subfolder named templates and place index.html file there.

Running the Machine Learning Model:
Open a command line interface and navigate to the project folder.
Run the script “receipts.py” with the command: “python receipts.py”
This trains the machine learning model and prepares it for predictions.

Starting the Flask Web Application:
In the same command line interface, run your Flask application script (e.g., app.py) using: “python app.py”.
Open a web browser and go to “http://127.0.0.1:5000/” to access the application.

Using the Web Application:
Select a month from the dropdown menu and click "Predict" to view the predicted receipt count.
This developed application also displays visual data charts for better understanding.
