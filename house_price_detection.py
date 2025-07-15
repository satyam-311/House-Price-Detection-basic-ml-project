# %%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")

# Load the dataset
@st.cache_data
def load_data():
    path = r"C:\Users\Satyam Mishra\OneDrive\Desktop\Housing.csv"
    df = pd.read_csv(path)
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Show data shape
st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop target if nulls
df = df.dropna(subset=["price"], axis=0)

# Select features (auto-inferred)
target = "price"
features = df.drop(columns=[target]).columns.tolist()

# Simple feature selection: numeric + object
num_features = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = df[features].select_dtypes(include=["object"]).columns.tolist()

# Train-test split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Final pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)
st.success("✅ Model trained successfully!")

# Evaluate
score = model.score(X_test, y_test)
st.metric("R² Score on Test Data", f"{score:.3f}")

# --- Prediction Section ---
st.header("🔮 Predict House Price")

user_input = {}
for col in num_features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    default_val = float(df[col].median())
    user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val)

for col in cat_features:
    user_input[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {prediction:,.0f}")


# %%



