# 🌦️ Rainfall Prediction System – Polynomial Ridge Regression (Mangalore Weather)

## Introduction
This project is a beginner-friendly Machine Learning application designed to predict rainfall (in millimeters) based on weather parameters such as pressure, temperature, humidity, and wind speed.

The project began as a simple exploration of linear regression but gradually evolved into a **Polynomial Ridge Regression model**, giving improved accuracy and non-linear interpretability. It includes a **Tkinter GUI interface** that allows users to adjust weather parameters using sliders and instantly see the predicted rainfall.

This project was built step-by-step under a structured learning journey inspired by Andrew Ng’s Machine Learning course.

## Exploratory Data Analysis
EDA was performed using Python libraries like Pandas, NumPy, Matplotlib, and Seaborn. The key observations include:

- **t2m (Temperature)** and **rh (Humidity)** are strongly negatively correlated (-0.60).
- **u10** and **ws** show strong positive correlation (+0.64).
- **tp (Total Precipitation)** is highly non-linear with **rh**, with a sharp increase beyond 75% humidity.
- Distributions:
  - `rh` is left-skewed.
  - `tp` is right-skewed (typical of rainfall data).
  - `sst` showed bimodal distribution.

Boxplots, histograms, scatter plots, and a correlation heatmap were used to visualize patterns.

## Model Building
Several regression models were explored:

| Model | Description | R² (Test) | Remarks |
|--------|--------------|-----------|----------|
| Linear Regression | Basic model | 0.44 | Simple, interpretable |
| Ridge Regression | L2 regularization | 0.44 | Stable but similar |
| Lasso Regression | L1 regularization | 0.44 | Sparse but unchanged |
| Polynomial Ridge Regression | Degree = 2 | **0.52** | Captured non-linear humidity–rainfall relationship |

The Polynomial Ridge model was chosen for deployment due to its superior performance and realistic rainfall curve behavior.

## GUI Interface (Tkinter)
A simple desktop interface was built using Tkinter.  
The GUI allows users to:
- Adjust weather parameters using sliders.
- Predict rainfall using the trained Polynomial Ridge model.
- Reset all sliders to default values.


## Technologies Used
- **Python 3.11**
- **Libraries:**
  - Pandas, NumPy, Matplotlib, Seaborn
  - Scikit-learn
  - Joblib
  - Tkinter (GUI)


