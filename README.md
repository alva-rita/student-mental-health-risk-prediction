# Student Mental Health Risk Prediction Using Machine Learning

A comparative machine learning project that predicts **student depression** and **stress levels** using supervised learning models. The project compares **Logistic Regression, Decision Tree, and Random Forest** across two publicly available datasets, demonstrating how effective data preprocessing and feature engineering improve predictive performance while maintaining model interpretability.

Key Achievement: Logistic Regression achieved the highest predictive performance on both datasets, highlighting the importance of data quality over model complexity.


## Project Overview

Universities collect large amounts of student-related data, yet many lack systems capable of identifying students at risk of mental health challenges before they escalate. Early identification can support timely interventions and improve student wellbeing and academic success.

This project applies supervised machine learning techniques to predict two important mental health outcomes (depression and stress) using two publicly available student datasets. Three classification models were developed, trained and evaluated to determine which algorithm provides the best balance between predictive performance and interpretability.

The project demonstrates an end-to-end data science workflow, including data cleaning, feature engineering, exploratory data analysis, model development, robustness testing and result interpretation.

## Business Problem

Mental health issues such as stress and depression are increasingly affecting university students worldwide. While universities routinely collect academic and demographic information, these data are rarely leveraged to support proactive mental health interventions.

The objective of this project is to investigate whether machine learning can accurately identify students at risk using structured student data, enabling earlier intervention and more informed decision-making.

## Objectives

The primary objectives of this project were to:

- Develop machine learning models capable of predicting student **depression** and **stress** using structured student datasets.
- Compare the performance of **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers.
- Evaluate model performance using **Accuracy, Precision, Recall, F1-score, and Confusion Matrices**.
- Assess model robustness under real-world data conditions, including missing values, outliers, and multicollinearity.
- Identify the most influential factors associated with student mental health outcomes.
- Demonstrate an end-to-end data science workflow from data preprocessing to model evaluation.

##  Repository Structure

```text
student-mental-health-risk-prediction/
│
├── notebooks/
│   ├── 01_depression_risk_analysis.ipynb
│   └── 02_stress_prediction_analysis.ipynb
│
├── report/
│   └── Rita_Nicholas_MSc_Dissertation.pdf
│
├── presentation/
│   └── Project_Presentation.pptx
│
├── images/
│   └── (Project screenshots and visualisations)
│
├── data/
│   └── README.md
│
└── README.md
```

## Technology Stack

| Category | Tools & Libraries |
|----------|-------------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook |
| Data Preparation | Microsoft Excel |
| Version Control | Git & GitHub |

## Project Workflow

The project followed a structured end-to-end machine learning workflow:

```text
Dataset Selection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Model Training & Comparison
        │
        ▼
Model Evaluation
        │
        ▼
Robustness Testing
        │
        ▼
Result Interpretation
```

## Dataset Description

Two publicly available student mental health datasets obtained from Kaggle were used in this project. Rather than focusing on a single mental health outcome, the study adopted a comparative approach by analysing **depression** and **stress** independently. This enabled the evaluation of machine learning models across two different classification problems while identifying differences in the factors influencing each condition.

| Dataset | Records | Features | Target Variable | Classification Type |
|----------|--------:|---------:|-----------------|---------------------|
| Student Depression Dataset | 27,901 | 17 | Depression | Binary Classification |
| Student Stress Dataset | 1,000 | 21 | Stress Level | Multi-class Classification (Low, Medium, High) |

The datasets were selected because they:
- contain psychological, academic, physiological and social variables relevant to student wellbeing;
- provide clearly defined target variables suitable for supervised learning;
- are publicly available, supporting transparency and reproducibility; and
- allow comparison of model performance across different mental health outcomes.

## Data Preprocessing

A structured preprocessing pipeline was implemented to improve data quality, reduce noise and prepare both datasets for machine learning.

Key preprocessing activities included:

- Removal of non-predictive variables such as **ID**.
- Evaluation of high-cardinality variables (**City**, **Profession**, and **Degree**) before grouping them into meaningful categories rather than removing them completely.
- Binary encoding of categorical variables including **Gender**, **Suicidal Thoughts**, and **Family History of Mental Illness**.
- Ordinal encoding of variables such as **Dietary Habits**.
- Feature engineering through grouping of cities, professions and degree fields.
- Discretisation of numerical variables including **Age**, **CGPA**, and **Sleep Duration**.
- Evaluation of low-variance variables before deciding whether to retain or remove them.
- Correlation analysis to better understand relationships between predictor variables and target outcomes.

These preprocessing steps produced structured datasets that improved model performance while maintaining interpretability.

## Model Training

Three supervised machine learning algorithms were implemented and evaluated:

| Model | Purpose |
|--------|---------|
| Logistic Regression | Baseline linear classification model |
| Decision Tree | Non-linear model providing rule-based decision making |
| Random Forest | Ensemble learning model capable of capturing complex feature interactions |

All models were trained using the same preprocessing pipeline and identical train-test split to ensure a fair comparison. Model performance was evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrices**.

To assess real-world applicability, robustness testing was also performed by introducing missing values, outliers and multicollinearity into the datasets.

## Results

Three supervised machine learning models were evaluated on both datasets using consistent training and evaluation procedures.

### Model Performance

| Model | Stress Prediction | Depression Prediction |
|--------|:-----------------:|:---------------------:|
| Logistic Regression | **91.5%** | **83.3%** |
| Decision Tree | **89%** | **75.7%** |
| Random Forest | **88.5%** | **82.6** |

### Evaluation Metrics

Model performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Across both datasets, **Logistic Regression consistently achieved the highest overall predictive performance**, outperforming the more complex Decision Tree and Random Forest models.

## Key Findings

The project produced several important findings:

- **Logistic Regression achieved the highest predictive performance** across both datasets, demonstrating that well-structured data can outperform increased model complexity.

- **Data preprocessing and feature engineering played a significant role** in improving model performance by reducing noise and creating more meaningful predictor variables.

- **Stress and depression exhibited different predictive patterns.** Stress was strongly associated with factors such as future career concerns, bullying and anxiety, while depression showed weaker but more distributed relationships across psychological, academic and lifestyle variables.

- **Model robustness testing demonstrated stable performance** under missing values, outliers and multicollinearity, suggesting potential applicability to real-world educational datasets.

- The findings reinforce that **interpretability is particularly valuable in mental health prediction**, where transparent models may be preferable to highly complex algorithms.

## Skills Demonstrated

This project demonstrates practical experience across the complete data science lifecycle.

| Skill | Application |
|--------|-------------|
| Data Cleaning | Removal of irrelevant variables, handling missing values and reducing noise |
| Feature Engineering | Encoding, grouping and discretisation of variables |
| Exploratory Data Analysis | Correlation analysis and visualisation using heatmaps |
| Machine Learning | Logistic Regression, Decision Tree and Random Forest |
| Model Evaluation | Accuracy, Precision, Recall, F1-score and Confusion Matrix |
| Robustness Testing | Evaluation under missing values, outliers and multicollinearity |
| Data Visualisation | Matplotlib and Seaborn |
| Technical Communication | Dissertation, presentation and project documentation |


## How to Run the Project

### Prerequisites

Ensure the following software is installed:

- Python 3.11 or later
- Jupyter Notebook (or JupyterLab)
- Git (optional, for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/alva-rita/student-mental-health-risk-prediction.git
cd student-mental-health-risk-prediction
```

### Install Required Libraries

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Alternatively, install the libraries manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Dataset

The datasets used in this project were obtained from Kaggle and are **not included** in this repository.

Download the datasets and place them in the appropriate project directory before running the notebooks.

### Run the Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open either notebook:

- `01_depression_risk_analysis.ipynb`
- `02_stress_prediction_analysis.ipynb`

Run the notebook cells sequentially to reproduce the preprocessing, model training, evaluation, and visualisations.

## Future Improvements

While the project achieved strong predictive performance, several opportunities exist for future enhancement:

- Evaluate additional machine learning models such as XGBoost, LightGBM and deep learning architectures.
- Incorporate behavioural indicators such as attendance, learning management system activity and library usage to improve predictive capability.
- Validate the models using data collected from UK higher education institutions to improve generalisability.
- Develop an interactive web application using Streamlit for real-time prediction and visualisation.
- Investigate model explainability techniques such as SHAP or LIME to provide greater transparency into prediction decisions.

## About the Author

**Rita Nicholas**

Product Manager transitioning into Data Analytics and Machine Learning, with experience delivering digital products within the fintech industry. Passionate about applying data-driven solutions to solve real-world problems across healthcare, education and financial technology.

- 💼 LinkedIn: *linkedin.com/in/rita-nicholas*
- 💻 GitHub: https://github.com/alva-rita

## References

- Kaggle – Student Depression Dataset
- Kaggle – Student Stress Monitoring Dataset
- Scikit-learn Documentation
- The Elements of Statistical Learning – Hastie, Tibshirani & Friedman
