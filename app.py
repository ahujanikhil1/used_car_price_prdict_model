'''from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import datetime
import os

app = Flask(__name__)

# Load CSV data
df = pd.read_csv("motorpedia-cars-data.cardekho-raw-cars-2024.csv")

# Clean column names
df.rename(columns={
    "make_year": "registration_year",
    "price": "selling_price",
    "owners": "owner"
}, inplace=True)

df.dropna(subset=["make", "model", "variant", "km_driven", "registration_year",
                  "owner", "fuel_type", "transmission", "selling_price"], inplace=True)

# Feature Engineering
current_year = datetime.datetime.now().year
df["car_age"] = current_year - df["registration_year"]
df = df[(df["km_driven"] < 500000) & (df["selling_price"] < 1e8)]

X = df[["make", "model", "variant", "fuel_type", "transmission", "owner", "km_driven", "car_age"]]
y = df["selling_price"]

# Preprocessing pipeline
categorical = ["make", "model", "variant", "fuel_type", "transmission", "owner"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline(steps=[
    ("pre", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
])

model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    makes = sorted(df["make"].unique())
    models = df.groupby("make")["model"].unique().apply(list).to_dict()

    # Convert tuple keys to strings for JSON serialization
    variants_raw = df.groupby(["make", "model"])["variant"].unique().apply(list).to_dict()
    variants = {f"{k[0]}|||{k[1]}": v for k, v in variants_raw.items()}

    if request.method == "POST":
        input_data = {
            "make": request.form["make"],
            "model": request.form["model"],
            "variant": request.form["variant"],
            "fuel_type": request.form["fuel_type"],
            "transmission": request.form["transmission"],
            "owner": request.form["owner"],
            "km_driven": int(request.form["km_driven"]),
            "car_age": current_year - int(request.form["registration_year"])
        }

        input_df = pd.DataFrame([input_data])
        predicted_price = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction=round(predicted_price, 2),
            form_data=input_data,
            makes=makes,
            models=models,
            variants=variants,
            current_year=current_year
        )

    return render_template(
        "index.html",
        prediction=None,
        makes=makes,
        models=models,
        variants=variants,
        current_year=current_year
    )

if __name__ == "__main__":
    app.run(debug=True)'''
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import datetime
import os

app = Flask(__name__)

# Load CSV data
df = pd.read_csv("motorpedia-cars-data.cardekho-raw-cars-2024.csv")

# Clean column names
df.rename(columns={
    "make_year": "registration_year",
    "price": "selling_price",
    "owners": "owner"
}, inplace=True)

# Drop missing values
df.dropna(subset=["make", "model", "variant", "km_driven", "registration_year",
                  "owner", "fuel_type", "transmission", "selling_price"], inplace=True)

# Feature Engineering
current_year = datetime.datetime.now().year
df["car_age"] = current_year - df["registration_year"]
df = df[(df["km_driven"] < 500000) & (df["selling_price"] < 1e8)]

X = df[["make", "model", "variant", "fuel_type", "transmission", "owner", "km_driven", "car_age"]]
y = df["selling_price"]

# Preprocessing pipeline
categorical = ["make", "model", "variant", "fuel_type", "transmission", "owner"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline(steps=[
    ("pre", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
])

model.fit(X, y)

def evaluate_car_condition_by_year_and_km(reg_year, km_driven, predicted_price):
    """
    Assign condition and price multiplier strictly based on registration year (2008-2025).
    Newer year = better condition and higher price multiplier.
    Kilometer ranges decrease price multiplier as km increases.
    """
    # Clamp reg_year to 2008-2025 range
    if reg_year < 2008:
        reg_year = 2008
    if reg_year > 2025:
        reg_year = 2025

    # Condition labels ordered from oldest (2008) to newest (2025)
    conditions_list = [
        "Old", "Very Old", "Old", "Older", "Older", "Fair", "Fair",
        "Average", "Average", "Good", "Good", "Better", "Better",
        "Very Good", "Very Good", "Excellent", "Excellent", "Best"
    ]
    year_index = reg_year - 2008
    condition = conditions_list[year_index]

    # Linear multiplier from 0.5 (2008) to 1.2 (2025)
    min_multiplier = 0.5
    max_multiplier = 1.2
    total_years = 2025 - 2008
    year_multiplier = min_multiplier + ((year_index / total_years) * (max_multiplier - min_multiplier))

    # Kilometer brackets with respective multipliers
    km_brackets = [
        (0, 5000, 1.0),
        (5001, 10000, 0.95),
        (10001, 20000, 0.9),
        (20001, 30000, 0.85),
        (30001, 40000, 0.8),
        (40001, 50000, 0.75),
        (50001, 60000, 0.7),
        (60001, 70000, 0.65),
        (70001, 80000, 0.6),
        (80001, 90000, 0.55),
        (90001, 100000, 0.5),
    ]

    km_multiplier = 0.5  # Default if km > 100000
    for low, high, mult in km_brackets:
        if low <= km_driven <= high:
            km_multiplier = mult
            break

    final_multiplier = year_multiplier * km_multiplier
    adjusted_price = predicted_price * final_multiplier

    low_price = round(adjusted_price * 0.95, 2)
    high_price = round(adjusted_price * 1.05, 2)

    return condition, low_price, high_price

@app.route("/", methods=["GET", "POST"])
def index():
    makes = sorted(df["make"].unique())
    models = df.groupby("make")["model"].unique().apply(list).to_dict()
    variants_raw = df.groupby(["make", "model"])["variant"].unique().apply(list).to_dict()
    variants = {f"{k[0]}|||{k[1]}": v for k, v in variants_raw.items()}

    prediction = None
    price_range_low = None
    price_range_high = None
    condition = None
    form_data = {}

    if request.method == "POST":
        reg_year = int(request.form["registration_year"])
        km_driven = int(request.form["km_driven"])

        input_data = {
            "make": request.form["make"],
            "model": request.form["model"],
            "variant": request.form["variant"],
            "fuel_type": request.form["fuel_type"],
            "transmission": request.form["transmission"],
            "owner": request.form["owner"],
            "km_driven": km_driven,
            "car_age": current_year - reg_year
        }

        input_df = pd.DataFrame([input_data])
        predicted_price = model.predict(input_df)[0]

        condition, price_range_low, price_range_high = evaluate_car_condition_by_year_and_km(
            reg_year, km_driven, predicted_price
        )

        prediction = round(predicted_price, 2)

        return render_template(
            "index.html",
            prediction=prediction,
            price_range_low=price_range_low,
            price_range_high=price_range_high,
            condition=condition,
            form_data=input_data,
            makes=makes,
            models=models,
            variants=variants,
            current_year=current_year
        )

    return render_template(
        "index.html",
        prediction=prediction,
        price_range_low=price_range_low,
        price_range_high=price_range_high,
        condition=condition,
        form_data=form_data,
        makes=makes,
        models=models,
        variants=variants,
        current_year=current_year
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Defaults to 8080 if PORT not set
    app.run(host="0.0.0.0", port=port, debug=True)


