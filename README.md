# 🩺 Diabetes Prediction using Random Forest

## 📘 Overview
This project uses a Random Forest Classifier to predict whether a person is likely to have diabetes based on diagnostic health parameters such as glucose level, BMI, age, insulin, etc.

## 🚀 Features
- Data cleaning and preprocessing (handling missing values)
- Feature scaling using MinMaxScaler
- Hyperparameter tuning with GridSearchCV
- Model evaluation (Accuracy, Precision, Recall, AUC)
- Feature importance visualization

## 📊 Dataset
**Source:** [PIMA Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic influence |
| Age | Age in years |
| Outcome | 1 - Diabetic, 0 - Non-Diabetic |

## 🧠 Model
Random Forest Classifier tuned with:
- `n_estimators = 300`
- `max_depth = 10`
- `min_samples_split = 5`
- `min_samples_leaf = 2`
- `max_features = 'sqrt'`

## 📈 Results
- Accuracy: **~90%**
- High recall and precision for diabetic patients
- Top features: Glucose, BMI, Age, DiabetesPedigreeFunction

## 🧩 Technologies Used
- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## 💾 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/Diabetes_Prediction_RandomForest.git
