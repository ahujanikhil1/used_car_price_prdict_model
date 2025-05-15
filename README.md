# 🚗 Used Car Price Prediction App

A Flask-based web application that predicts the **selling price of a used car** using a machine learning model trained on real-world car listings from the Motorpedia (Cardekho) dataset.

---

## 🌟 Features

- 🔍 Predicts car prices using XGBoost regression
- 📋 Inputs: Make, Model, Variant, Fuel Type, Transmission, Owner Type, KM Driven, Registration Year
- 🧠 Estimates car **condition** (Good, Intermediate, Average, Poor)
- 💰 Displays predicted **price** and an **estimated price range**
- 🖥️ Simple, interactive HTML form interface

---

## 🧠 Machine Learning Overview

- **Algorithm**: XGBoost Regressor
- **Preprocessing**: OneHotEncoding for categorical features
- **Features Used**:
  - Make, Model, Variant
  - Fuel Type, Transmission
  - Owner Type, Kilometers Driven, Car Age
- **Training Data**: `motorpedia-cars-data.cardekho-raw-cars-2024.csv`

---

## 📁 Project Structure
car-price-predictor/
├── app.py # Main Flask app with ML model
├── motorpedia-cars-data.cardekho-raw-cars-2024.csv # Used car data
├── requirements.txt # Python dependencies
└── templates/
└── index.html # HTML form (must be created)

