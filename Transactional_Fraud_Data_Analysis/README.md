# Transactional Fraud Data Analysis

A machine learning project focused on detecting fraudulent financial transactions using data preprocessing, feature engineering, imbalance handling, and classification models.

---

## Project Overview

This project analyzes transactional data to identify fraudulent activities using supervised machine learning techniques. The workflow includes:

* Data preprocessing and cleaning
* Handling missing values
* Feature engineering
* Encoding categorical variables
* Handling class imbalance using SMOTE
* Model training and evaluation
* Hyperparameter tuning
* ROC-AUC performance analysis

The primary objective is to build an effective fraud detection model capable of distinguishing fraudulent and non-fraudulent transactions with high accuracy and recall.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## Project Structure

```text
Transactional-Fraud-Analysis/
│
├── Transactional Fraud Data Analysis.ipynb
├── README.md
└── dataset/
```

---

## Workflow

### 1. Import Libraries & Load Data

* Imported required Python libraries
* Loaded transactional dataset
* Performed initial exploratory analysis

### 2. Data Preprocessing

* Handled missing values
* Cleaned inconsistent records
* Processed datetime-related features

### 3. Feature Engineering

Created additional features based on transaction patterns and concurrency behavior.

### 4. Encoding Categorical Variables

Applied:

* Label Encoding
* Target Encoding

for categorical attributes.

### 5. Handling Imbalanced Data

Used SMOTE (Synthetic Minority Oversampling Technique) to balance fraudulent and non-fraudulent classes.

### 6. Model Training & Evaluation

Trained multiple machine learning models and evaluated them using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC Score

### 7. Hyperparameter Tuning

Optimized the best-performing model to improve prediction performance.

---

## Machine Learning Models Used

Some of the evaluated models include:

* Random Forest Classifier
* Logistic Regression
* Decision Tree Classifier
* Other comparative models

The final selected model demonstrated strong fraud detection performance after tuning and validation.

---

## Evaluation Metrics

The project uses the following evaluation metrics:

* Confusion Matrix
* ROC Curve
* ROC-AUC Score
* Precision & Recall
* F1 Score

---

## Key Learnings

* Importance of preprocessing in fraud detection
* Impact of class imbalance on model performance
* Benefits of feature engineering
* Model comparison and optimization techniques
* Real-world fraud analytics workflow

---

## Future Improvements

* Deploy the model using Flask or FastAPI
* Build an interactive Power BI or Streamlit dashboard
* Implement real-time fraud prediction
* Experiment with deep learning models
* Add advanced anomaly detection techniques

---

## Author

**Isha Prajapati**
Data Analyst | Machine Learning Enthusiast

---

## License

This project is for educational and academic purposes.
