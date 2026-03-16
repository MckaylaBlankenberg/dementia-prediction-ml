# Dementia Prediction using Machine Learning

This project demonstrates an end-to-end machine learning pipeline to predict dementia diagnosis using patient health data.  
It covers **data preprocessing, feature selection, model training, and evaluation**.

---

## 🔧 Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn (Logistic Regression, Decision Tree, SelectKBest, RFE)  

---

## 📊 Workflow

### 1. Data Preprocessing
- Removed empty/unnecessary columns.  
- Handled missing values (e.g., `EducationLevel` column with 446 nulls).  
- Encoded categorical variables using `pd.get_dummies`.  
- Standardized features with `StandardScaler`.

### 2. Feature Selection
- **SelectKBest (Filter-based)**: Selected top 10 features using ANOVA F-test.  
- **Recursive Feature Elimination (Wrapper-based)**: Used RandomForestClassifier to iteratively select the best 10 features.

### 3. Model Training
- **Logistic Regression**: Trained using both SelectKBest and RFE-selected features.  
- **Decision Tree Classifier**: Trained using both feature selection methods for comparison.

### 4. Model Evaluation
- Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score.  
- Logistic Regression: ~78–79% accuracy.  
- Decision Tree: ~96% accuracy, strong precision and recall.  
- Highlighted importance of **recall** in medical datasets to minimize false negatives.

---

## ✅ Results
| Model | Feature Selection | Accuracy | Key Notes |
|-------|------------------|----------|-----------|
| Logistic Regression | SelectKBest | ~79% | Balanced precision/recall |
| Logistic Regression | RFE | ~78% | Stable but slightly lower recall |
| Decision Tree | SelectKBest | ~96% | High precision & recall |
| Decision Tree | RFE | ~96% | Nearly identical performance |

---

## 🚀 How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/Kayla1RB/dementia-prediction-ml.git
