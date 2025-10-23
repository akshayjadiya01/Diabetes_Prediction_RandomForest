# 🩺 Diabetes Prediction using Decision Tree and Random Forest

### 📘 Overview
This project focuses on predicting the likelihood of diabetes in patients based on health diagnostic data using **Decision Tree** and **Random Forest** classifiers.  
The model leverages data preprocessing, feature scaling, hyperparameter tuning, and performance visualization to create a reliable prediction system.  
It demonstrates a strong understanding of **Machine Learning concepts**, **model evaluation**, and **data visualization** in Python.

---

## 🚀 Key Highlights
- Implemented **Decision Tree** and **Random Forest Classifiers**
- Applied **MinMaxScaler** for feature normalization
- Performed **Hyperparameter Tuning (GridSearchCV)** for Random Forest
- Compared model performance using **Accuracy**, **AUC**, and **Confusion Matrix**
- Visualized model insights with:
  - Correlation heatmap
  - ROC curves
  - Feature importance bar charts
  - Decision tree diagram

---

## 📊 Dataset
**Source:** [PIMA Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic influence |
| Age | Age in years |
| Outcome | 1 = Diabetic, 0 = Non-Diabetic |

---

## 🧠 Model Development Steps
### 1️⃣ Data Preprocessing
- Loaded the PIMA Indians Diabetes Dataset.
- Checked for missing values and performed exploratory data analysis (EDA).
- Scaled all numeric features using **MinMaxScaler** for uniform distribution.

### 2️⃣ Decision Tree Classifier
- Implemented with Gini Index and `max_depth=5`.
- Visualized the tree structure using `plot_tree()`.
- Evaluated with **accuracy, confusion matrix, ROC curve**, and **classification report**.

### 3️⃣ Random Forest Classifier
- Applied **GridSearchCV** for hyperparameter tuning.
- Trained using best parameters for optimal performance.
- Generated **feature importance** visualization.

### 4️⃣ Model Evaluation
- Compared Decision Tree and Random Forest using:
  - Accuracy
  - AUC Score
  - ROC Curve
  - Confusion Matrix
- Random Forest outperformed Decision Tree with better generalization.

---

## 🧩 Technologies Used
| Technology | Purpose |
|-------------|----------|
| Python | Programming language |
| Pandas, NumPy | Data handling & manipulation |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Model building, evaluation, and tuning |
| Joblib | Model persistence |

---

## 📈 Results Summary
| Model | Accuracy | AUC Score |
|--------|-----------|-----------|
| Decision Tree | ~85% | 0.87 |
| Random Forest | ~90% | 0.93 |

🎯 **Random Forest achieved the best performance**, highlighting its ensemble strength and reduced overfitting.

---

## 📊 Visualizations
- 📉 Correlation Heatmap  
- 🌲 Decision Tree Plot  
- 🧾 Confusion Matrices (DT & RF)  
- 📊 Feature Importance Chart  
- 🔵 ROC Curve Comparison

---

## 💾 Model Saving
Both models are saved as `.pkl` files for easy deployment:
rf_model.pkl # Random Forest Model
dt_model.pkl # Decision Tree Model


---

## 🧪 How to Run the Project

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/Diabetes_Prediction_DecisionTree_RandomForest.git
cd Diabetes_Prediction_DecisionTree_RandomForest
```




### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run the Notebook

jupyter notebook Diabetes_Prediction_DecisionTree_RandomForest.ipynb

---
