# 🏠 House Price Detection Web App

This project is a simple machine learning web application that predicts house prices based on user-input features such as number of rooms, area, and other property details.

Built using **Python**, **Streamlit**, **Scikit-learn**, and **Pandas**, the app provides an interactive interface where users can input housing features and instantly receive a price prediction.

The machine learning model used is a **Random Forest Regressor**, trained on a cleaned dataset (`Housing.csv`). Numerical values are preprocessed with median imputation, and categorical features are encoded using one-hot encoding. The model is evaluated using the R² score to ensure reasonable accuracy.

This app demonstrates how ML models can be integrated with simple front-end tools like Streamlit to create deployable data applications.
