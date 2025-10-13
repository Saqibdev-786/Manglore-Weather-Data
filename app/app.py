import tkinter as tk
import joblib
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(__file__))  # one level up from /app
model_path = os.path.join(base_dir, "models", "ridge_poly_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
poly_path = os.path.join(base_dir, "models", "poly_transformer.pkl")


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
poly = joblib.load(poly_path)


window=tk.Tk()
window.title("Mangalore Weather Data")
window.geometry("600x700")
window.config(bg="#87CEEB")

title_label = tk.Label(
    window,
    text="Rainfall Prediction System",
    font=("Ariel", 24, "bold"),
    bg="#E8F0FE",
    fg="#1F4E79"
)
title_label.pack(pady=20)  


result_label = tk.Label(
    window,
    text="Predicted Rainfall: -- mm",
    font=("Ariel", 18, "bold"),
    bg="#E8F0FE",
    fg="black"
)
result_label.pack(pady=10)


frame = tk.Frame(window, bg="#E8F0FE", padx=20, pady=20)
frame.pack(pady=20)

features = {
    "msl": (100000, 101500),
    "sst": (25, 32),
    "u10": (-10, 10),
    "v10": (-10, 10),
    "ws": (0, 20),
    "t2m": (24, 32),
    "rh": (40, 100)
}


sliders = {}


# Create label + slider for each feature
for i, (feature, (min_val, max_val)) in enumerate(features.items()):
    tk.Label(
        frame,
        text=feature.upper(),
        bg="#E8F0FE",
        font=("Georgia", 12, "bold"),
        fg="#1F4E79"
    ).grid(row=i, column=0, sticky="w", padx=10, pady=5)

    var=tk.DoubleVar()

    slider=tk.Scale(
        frame,
        from_=min_val,
        to=max_val,
        orient="horizontal",
        variable=var,
        resolution=0.1,
        length=300,
        bg="#E8F0FE",
        highlightthickness=0
    )
    slider.set((min_val + max_val) / 2)
    slider.grid(row=i, column=1, padx=10, pady=5)

    sliders[feature] = var


def predict_rainfall():
    input_data = np.array([var.get() for var in sliders.values()]).reshape(1, -1)
    
    input_data_scaled = scaler.transform(input_data)

    input_data_poly = poly.transform(input_data_scaled)

    prediction = model.predict(input_data_poly)[0]

    result_label.config(text=f"Predicted Rainfall: {prediction:.2f} mm")


predict_button = tk.Button(
    window,
    text="Predict Rainfall",
    font=("Ariel", 15, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=15,
    pady=10,
    command=predict_rainfall
)
predict_button.pack(pady=20)

window.mainloop()
