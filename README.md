# Cuisine Classification

A full-stack Streamlit web application that predicts restaurant cuisines using a trained multi-label classification model. The project combines a sleek, responsive user interface with secure login functionality and a powerful prediction engine, delivering a seamless experience tailored for restaurant profiling, food-tech dashboards, and smart cuisine suggestions.

---

##  Project Overview

**Cuisine Classification** enables users to input restaurant-specific attributes—such as city, pricing, table booking options, delivery availability, and more—and receive a list of predicted cuisines with confidence scores. Designed with a focus on both visual finesse and technical robustness, this app showcases the integration of a machine learning model into a modern, user-centric interface.

## key features:

-  Secure registration and login system using local storage  
-  Cuisine prediction powered by a trained multi-label XGBoost classifier  
-  Clean, centered layout with a translucent UI and elegant background styling  
-  Adjustable confidence thresholds for prediction sensitivity  
-  Dynamic city list loaded from data for intelligent auto-completion

---



## Installation & Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

```bash
pip install streamlit pandas numpy joblib scikit-learn xgboost
```

2. **Run the application:**

```bash
python -m streamlit run app.py
```

> The app will open in your browser automatically.

---

## How It Works

The model takes structured restaurant features including:

- City
- Average cost for two
- Price range
- Table booking availability
- Online delivery
- Votes and ratings

These features are passed to a trained XGBoost model wrapped in a `MultiOutputClassifier`, which predicts cuisines such as North Indian, Chinese, Fast Food, etc., with probabilities. Only predictions above the user-defined confidence threshold are shown.

---

## Technologies Used

- Streamlit
- pandas, numpy
- scikit-learn
- xgboost
- joblib (for model persistence)

---

##  Security Notes

- For demonstration purposes, passwords are stored in plaintext in `users.json`.  
  In production, you should hash passwords using `bcrypt`.

---

##  Future Enhancements

-  Password encryption for secure storage  
-  User-specific prediction logs  
-  Dashboard analytics per cuisine/city  
-  Expand dataset with restaurant names and cuisines

---

