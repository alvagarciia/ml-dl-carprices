# 🚗 Car Price Prediction Web App

## 📌 Project Overview

This web application predicts the market price of used cars based on various features such as mileage, brand, model, fuel type, and more. Built using Python and Streamlit, the app leverages machine learning models to provide pricing estimates, assisting both buyers and sellers in making informed decisions.

## 🧠 Technologies Used

- **Python 3.10**
- **Streamlit** – For the interactive web interface
- **scikit-learn** – For ML modeling and preprocessing
- **XGBoost** – For advanced gradient boosting
- **joblib** – For model serialization
- **category_encoders** – For categorical encoding

## 🚀 Features

- 📝 User input form for entering car specifications
- 🤖 Pre-trained ML and DL models for price prediction
- 🧼 Built-in preprocessing (encoding, imputation, etc.)
- 💾 Serialized models using `joblib` and Keras `.h5` or `.keras` files

## 🛠️ Installation & Setup

### Prerequisites

Ensure Python 3.10 is installed. It's recommended to use a virtual environment (e.g., via `conda` or `venv`).

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   ```

2. Install dependencies:

    ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

    **Standard usage:**
    ```bash
   streamlit run app.py
   ```

    **If you're using WSL (Windows Subsystem for Linux):**
    ```bash
   streamlit run app.py --server.headless true --server.enableCORS false --server.address=0.0.0.0
   ```
   This allows you to access the app from your Windows browser when running Streamlit inside WSL.

4. Open the provided local URL in your browser to access the app. 

## 🧪 Model Training

The ML and DL models were trained on a Kaggle dataset of used cars. Training included:

- Data cleaning and preprocessing
- Feature engineering (e.g., computing car age)
- Training an `XGBRegressor` and a Keras-based neural network
- Evaluating using cross-validation and mean absolute error (MAE)
- Saving the trained models for deployment

## ⚠️ Notes

- **Version Warning**: The saved models were trained using `scikit-learn==1.2.2`. Loading them in newer versions (e.g., 1.7.0) may show warnings. This is acceptable for portfolio purposes but should be addressed for production use.
- **Dataset File**: The original dataset used for training is not required to use the app, as the models are already trained and serialized.

## 📸 Screenshot

> _Optional: Add a screenshot of your Streamlit app here_  
> Example:

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.