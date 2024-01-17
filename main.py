# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(file_path):
    # Load historical stock price data
    # Assuming the data is in a CSV file with columns 'Date' and 'Close'
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Extract features (X) and target variable (y)
    X = np.array(data.index).reshape(-1, 1)  # Using index as a feature (you may use other features)
    y = data['Close'].values
    return X, y

def train_model(X_train, y_train):
    # Create a linear regression model
    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    # Make predictions on the test set
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def display_metrics(mse, r2):
    # Display performance metrics
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

def plot_results(X_test, y_test, y_pred):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Actual Prices')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def predict_new_data(model, new_data):
    # Make predictions for new data
    new_X = np.array(new_data.index).reshape(-1, 1)
    new_pred = model.predict(new_X)
    return new_X, new_pred

def main():
    # Specify the file path
    file_path = 'path/to/your/data.csv'

    # Load data
    data = load_data(file_path)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = make_predictions(model, X_test)

    # Evaluate and display metrics
    mse, r2 = evaluate_model(y_test, y_pred)
    display_metrics(mse, r2)

    # Plot the results
    plot_results(X_test, y_test, y_pred)

    # Predict for new data (example)
    new_data = pd.DataFrame({'Date': ['2024-01-18', '2024-01-19'], 'Close': [0, 0]})
    new_X, new_pred = predict_new_data(model, new_data)
    print(f'Predictions for new data:\n{pd.DataFrame({"Date": new_data["Date"], "Predicted Close": new_pred})}')

if __name__ == "__main__":
    main()
