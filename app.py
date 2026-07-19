from flask import Flask, render_template, request
from predict import predict_price

import pandas as pd
from config import DATA_PATH

app = Flask(__name__)

# ==========================
# Load dataset 1 lần
# ==========================

df = pd.read_csv(DATA_PATH)

brands = sorted(df["Brand"].unique())

engine_sizes = sorted(df["Engine_Size"].unique())

fuel_types = sorted(df["Fuel_Type"].unique())

transmissions = sorted(df["Transmission"].unique())

brand_models = {}

for brand in brands:
    brand_models[brand] = sorted(
        df[df["Brand"] == brand]["Model"].unique()
    )


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        user_input = {

            "Brand": form_data["Brand"],

            "Model": form_data["Model"],

            "Year": int(form_data["Year"]),

            "Engine_Size": float(form_data["Engine_Size"]),

            "Fuel_Type": form_data["Fuel_Type"],

            "Transmission": form_data["Transmission"],

            "Mileage": float(form_data["Mileage"]),

            "Doors": int(form_data["Doors"]),

            "Owner_Count": int(form_data["Owner_Count"])

        }

        prediction = int(round(predict_price(user_input), 0))

    return render_template(

        "index.html",

        prediction=prediction,

        brands=brands,

        engine_sizes=engine_sizes,

        fuel_types=fuel_types,

        transmissions=transmissions,

        brand_models=brand_models,

        form=form_data

    )


if __name__ == "__main__":
    app.run(debug=True)
