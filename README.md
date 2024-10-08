## Stock Prediction (ML project)

**Getting Started:**

1. **Initialize a virtual environment:**

   ```bash
   python3 -m venv venv 
2. **Install dependencies**

    ```bash
   pip install -r requirements.txt

**About the project:**

This project utilizes various Python libraries to build a stock prediction model. The libraries used include:

* **NumPy:** For numerical operations and array manipulation.
* **Pandas:** For data manipulation and analysis. Used for loading and cleaning stock data.
* **yfinance:** For downloading historical stock data from Yahoo Finance.
* **Keras:** High-level API for building and training neural networks.
* **keras-models:** Provides pre-trained models and building blocks for Keras.
* **Streamlit:** For building interactive web applications for data science and machine learning. Used to create a user interface for the prediction model.
* **Matplotlib:** For data visualization. Used to plot stock prices and model predictions.
* **Scikit-learn:** For machine learning tasks such as data preprocessing, model selection, and evaluation.
* **TensorFlow:** Open-source library for numerical computation and large-scale machine learning. Used as the backend for Keras.

**Project Workflow:**

1. **Data Acquisition:** Download historical stock data using yfinance.
2. **Data Preprocessing:** Clean and prepare the data using Pandas and Scikit-learn. This includes handling missing values, scaling data, and feature engineering.
3. **Model Building:** Build a prediction model using Keras and TensorFlow. This could involve using a recurrent neural network (RNN) like LSTM or GRU.
4. **Model Training:** Train the model on the historical stock data.
5. **Model Evaluation:** Evaluate the model's performance using metrics like mean squared error (MSE) or root mean squared error (RMSE).
6. **Deployment:** Deploy the model using Streamlit to create an interactive web application for users to input stock symbols and get predictions.

**Note:**

Stock prediction is a complex task and achieving high accuracy is challenging. This project is intended for educational purposes and should not be used for making investment decisions.
