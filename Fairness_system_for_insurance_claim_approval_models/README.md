# Fairness-Aware Insurance Claim Approval System

## Project Overview

This project develops a fairness-aware machine learning system for insurance claim approval prediction. The objective is not only to build predictive models, but also to evaluate, explain, and mitigate potential bias within automated decision-making systems.

The project combines:
- Machine Learning Classification
- Fairness Auditing
- Explainable AI (XAI)
- Bias Mitigation
- Ethical AI Analysis

to create a more transparent and responsible insurance approval framework.

---

## Objectives

- Predict insurance claim approval outcomes using machine learning
- Perform exploratory data analysis and feature engineering
- Compare multiple classification models
- Evaluate model fairness across demographic groups
- Detect and analyze bias using fairness metrics
- Apply explainability techniques using SHAP
- Discuss ethical implications of automated insurance decision systems

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

---

## Machine Learning Models Used

The following classification models were implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Hyperparameter tuning was performed using `GridSearchCV`.

---

## Dataset Features

The dataset includes demographic, financial, and claim-related information such as:

- Claim Amount
- Patient Age
- Patient Income
- Patient Gender
- Provider Specialty
- Claim Type
- Employment Status
- Marital Status
- Provider Location
- Claim Submission Method

Additional engineered features were created to improve predictive capability.

---

## Feature Engineering

The following feature engineering techniques were applied:

- Claim-to-income ratio
- Provider claim frequency
- Patient claim frequency
- Age-income interaction features
- Age groups
- Claim size categorization
- One-hot encoding for categorical variables

---

## Fairness Auditing Metrics

The project evaluates fairness using multiple Responsible AI metrics:

- Demographic Parity Difference
- Equal Opportunity Difference
- Equalized Odds Difference
- Disparate Impact Ratio

These metrics were used to assess bias across gender groups within claim approval predictions.

---

## Bias Mitigation

Bias mitigation strategies were explored through threshold adjustment techniques and fairness-aware evaluation.

The project compares fairness metrics:
- Before mitigation
- After mitigation

to analyze fairness-performance tradeoffs.

---

## Explainable AI (XAI)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and improve transparency.

The project includes:
- SHAP Beeswarm Plot
- SHAP Waterfall Plot
- Feature Importance Analysis

These techniques help explain how features influence claim approval decisions.

---

## Evaluation Metrics

Classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Fairness metrics were evaluated separately to assess ethical model performance.

---

## Visualizations Included

The notebook contains multiple visualizations including:

- Approval distribution analysis
- Correlation heatmaps
- ROC Curve
- Confusion Matrix
- Feature Importance Plot
- SHAP Explainability Plots
- Fairness Comparison Charts
- Approval Rate Comparisons

---

## Ethical Considerations

The project discusses:
- Bias in automated insurance systems
- Fairness-performance tradeoffs
- Risks of demographic discrimination
- Importance of explainable AI
- Responsible AI deployment considerations

---

## Key Findings

- Ensemble models achieved the strongest predictive performance
- Fairness metrics revealed measurable demographic disparities
- Explainability techniques improved transparency
- Bias mitigation slightly improved fairness metrics while affecting predictive performance


---

## Conclusion

This project demonstrates a complete Responsible AI workflow by integrating machine learning, fairness auditing, explainability, and ethical analysis into insurance claim approval prediction.

The notebook highlights the importance of balancing predictive performance with fairness, transparency, and accountability in high-stakes automated decision systems.

---

## Author

Isha Prajapati
