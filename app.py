'''from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import datetime

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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import datetime

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

        km = input_data["km_driven"]
        age = input_data["car_age"]

        # Car condition logic
        if km < 50000 and age < 5:
            condition = "Good"
            price_range_low = predicted_price * 0.9
            price_range_high = predicted_price * 1.1
        elif 50000 <= km <= 75000 and 5 <= age <= 8:
            condition = "Intermediate"
            price_range_low = predicted_price * 0.8
            price_range_high = predicted_price * 1.0
        elif km > 75000 and age > 8:
            condition = "Poor"
            price_range_low = predicted_price * 0.6
            price_range_high = predicted_price * 0.8
        else:
            condition = "Average"
            price_range_low = predicted_price * 0.7
            price_range_high = predicted_price * 0.9

        return render_template(
            "index.html",
            prediction=round(predicted_price, 2),
            price_range_low=round(price_range_low, 2),
            price_range_high=round(price_range_high, 2),
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

