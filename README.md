# 🏠 House Price Prediction Web App

This project is an end-to-end machine learning solution designed to predict house prices based on user-provided property features. Built using **Streamlit** for the frontend and **Scikit-learn** for the backend model, this lightweight app allows real-time predictions through an interactive web interface.

## 🔍 Problem Statement

Predicting house prices is a classic regression problem with real-world significance. By analyzing historical housing data, the objective is to train a machine learning model capable of generalizing price predictions on unseen property features.

## 🧠 Solution Overview

The application is powered by a **Random Forest Regressor**, trained on a structured housing dataset. The pipeline includes:

- Handling of missing values
- One-hot encoding for categorical features
- Feature scaling (if required)
- Model evaluation using **R² score**

The model is integrated directly into a **Streamlit app**, enabling live inference based on user inputs. Users can adjust numerical and categorical features using sliders, dropdowns, and input boxes to receive dynamic predictions.

## 📦 Tech Stack

- **Python 3.13**
- **Pandas** for data manipulation
- **Scikit-learn** for model training
- **Streamlit** for UI & web deployment

## ⚙️ Use Cases

- Educational demo of regression deployment
- Starter template for real-estate pricing apps
- Quick MVP for property-related business logic

## ✨ Highlights

- Clean & responsive UI using Streamlit widgets
- Modular, readable code structure
- Reproducible results with consistent pipeline
- Dataset-driven insights for user input

## 📁 Dataset

The app uses a CSV file (`Housing.csv`) containing various housing-related attributes such as:

- Number of bedrooms/bathrooms
- Total area
- Construction year
- Sale price (target)

You can replace this with your own dataset by adjusting the input schema in the code.

---

This project is intended to demonstrate the simplicity of turning a trained machine learning model into an interactive web application — suitable for prototypes, learning purposes, or initial deployments.

