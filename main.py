# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('C:/Users/Zaiba Farheen/PycharmProjects/pythonProject2/DataScienceEL/Datasets/BreastCancerDataset.csv')

# Drop the 'id' column
df.drop('id', axis=1, inplace=True)

# Mapping 'M' with 1 and 'B' with 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define Feature Matrix (X) and Target Array (y)
X = np.array(df[['texture_mean', 'radius_mean']])
y = np.array(df['diagnosis'])

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input for prediction
        texture_mean = float(request.form['texture_mean'])
        radius_mean = float(request.form['radius_mean'])

        # Predict for new data
        new_data_point = np.array([[texture_mean, radius_mean]])
        result = knn.predict(new_data_point)

        if result[0] == 1:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

        return render_template('index.html', diagnosis=diagnosis)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
