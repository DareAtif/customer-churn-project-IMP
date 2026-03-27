#  Customer Churn Prediction (End-to-End ML Project)

## Project Overview

This project predicts whether a customer is likely to churn (leave the service) using Machine Learning.
It helps businesses (Telecom / SaaS) identify at-risk customers and take proactive retention actions.

---

## Business Problem

Customer churn leads to significant revenue loss.

👉 Goal:

* Identify customers likely to churn
* Reduce churn rate
* Improve customer retention

---

##  Business Impact

* Early churn prediction
* Targeted retention strategies
* Revenue loss reduction

# Example:
If 1000 customers churn and each pays ₹2000/month → Loss = ₹20,00,000
If model saves 30% customers → ₹6,00,000/month saved

---

##  Dataset

* **Telco Customer Churn Dataset**
* Contains customer demographics, services, billing details, and churn status

---

##  Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* XGBoost 
* Streamlit (for UI) future approach

---

##  Machine Learning Approach

* Data Cleaning & Preprocessing
* Feature Engineering
* Model Training
* Hyperparameter Tuning (GridSearchCV)
* Evaluation using Recall, Precision, F1-score

---

##  Models Used

* Logistic Regression
* Random Forest
* XGBoost (Best Model )

---

##  Key Features

✔ End-to-End Pipeline (Preprocessing + Model)
✔ Feature Importance Analysis
✔ Streamlit Web App
✔ Real-time Prediction
✔ Business Insights Output

---

##  Model Performance

* Optimized for **Recall** (important for churn detection)
* XGBoost performed best

---

##  Streamlit App Features

* Input customer details
* Predict churn probability
* Show risk level
* Provide business recommendations

---

##  Project Structure

```
customer-churn-project/
│
├── data/
├── src/
│   ├── train_pipeline.py
│   ├── predict_pipeline.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/customer-churn-project.git
cd customer-churn-project
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Train Model

```
python src/train_pipeline.py
```

### 4️⃣ Run Prediction

```
python src/predict_pipeline.py
```

### 5️⃣ Run Streamlit App

```
streamlit run app/app.py
```

---

## 🌍 Deployment

* Deployed using **Render**
* Also compatible with **Hugging Face Spaces**

---

##  Key Learnings

* Handling real-world business problems
* Feature engineering & preprocessing pipelines
* Model optimization using GridSearch
* Deployment of ML models

---

##  Future Improvements

* Add FastAPI backend
* Integrate database
* Add customer segmentation
* Email/SMS alerts for high-risk customers

---

##  Author

Sayed Atif Hosen

---

##  If you like this project

Give it a ⭐ on GitHub and share your feedback!
