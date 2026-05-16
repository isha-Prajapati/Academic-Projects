# Weather Forecasting Using Machine Learning and Time-Series Analysis

## Project Overview

This project focuses on forecasting daily maximum temperature (`maxtp`) using machine learning and time-series forecasting techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, time-series cross-validation, model comparison, recursive forecasting, and forecasting evaluation.

The project compares multiple forecasting models and evaluates their predictive performance using forecasting-specific metrics and temporal validation strategies.

---

## Objectives

- Perform time-series preprocessing and exploratory analysis
- Engineer lag-based and rolling statistical features
- Compare machine learning forecasting models
- Apply TimeSeriesSplit for temporal cross-validation
- Evaluate forecasting performance using multiple metrics
- Implement recursive multi-step forecasting
- Compare forecasting performance against baseline and ARIMA models

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

---

## Machine Learning Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- ARIMA (baseline statistical forecasting model)

---

## Forecasting Methodology

The project follows a structured forecasting workflow:

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Time-Series Cross Validation using `TimeSeriesSplit`
5. Model Training and Evaluation
6. Best Model Selection
7. Recursive Forecasting
8. Forecast Visualization and Interpretation

---

## Feature Engineering

The following forecasting features were created:

- Lag Features (`lag1`, `lag7`)
- Rolling Mean Features
- Month-based Temporal Features

These features help capture temporal dependencies and seasonality patterns in the weather data.

---

## Evaluation Metrics

Models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- SMAPE (Symmetric Mean Absolute Percentage Error)

SMAPE was used instead of traditional MAPE to provide more stable evaluation for near-zero target values.

---

## Validation Strategy

`TimeSeriesSplit` was used for temporal cross-validation to preserve chronological ordering and prevent future data leakage.

A final chronological train-test split was also used for final forecasting evaluation.

---

## Key Results

- Random Forest achieved the best forecasting performance
- The model outperformed both:
  - Naive forecasting baseline
  - ARIMA forecasting model
- Achieved strong forecasting accuracy with:
  - High R² Score
  - Low forecasting error
  - Stable SMAPE values

---

## Recursive Forecasting

Recursive forecasting was implemented for multi-step future prediction. Previous predictions were iteratively fed back into the model to generate future forecasts.

---

## Visualizations Included

- Temperature Trend Analysis
- Seasonal and Monthly Trends
- Correlation Heatmap
- Forecast vs Actual Visualization
- Forecast Performance Metrics
- Recursive Forecast Plots

---

## Limitations

- Recursive forecasting may accumulate forecasting error over longer horizons
- ARIMA model was not extensively tuned
- Confidence intervals were not implemented for future forecasts

---

## Conclusion

This project demonstrates a complete machine learning-based forecasting workflow using proper temporal validation, forecasting metrics, recursive forecasting, and comparative model evaluation.

The Random Forest forecasting model successfully captured nonlinear temporal patterns and achieved strong forecasting performance on unseen weather data.


---

## Author

Isha Prajapati
