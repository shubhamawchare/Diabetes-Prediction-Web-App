# 🩺Diabetes App

**A simple Streamlit app for exploring the Pima Indians Diabetes dataset, visualizing features, and training two basic classification models (Logistic Regression and Decision Tree).**

---

## 📋Table of contents

* [Project overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [How the app preprocesses data](#how-the-app-preprocesses-data)
* [Quick start — run locally](#quick-start---run-locally)
* [App pages & usage](#app-pages--usage)
* [Model details](#model-details)
* [Known limitations & notes](#known-limitations--notes)
* [Suggestions & next steps](#suggestions--next-steps)
* [Repository structure](#repository-structure)
* [License](#license)

---

## 📁Project overview

This repository contains a Streamlit application (`diabetes.py`) that loads a diabetes dataset (`diabetes.csv`), performs simple cleaning, offers interactive visualizations, and allows users to train and evaluate two machine learning models (Logistic Regression and Decision Tree) and try simple predictions.

The goal is educational: to demonstrate an end-to-end workflow for data exploration, model training, and quick prediction inside an interactive web UI.

> ⚠️ **Important:** This project is for learning and demonstration purposes only. It is **not** a clinical tool and must not be used for medical decision making.

---

## 💻Features

* Interactive Streamlit UI with a sidebar-driven navigation
* Home page with dataset preview and basic statistics
* Data visualizations: histograms, scatterplots, correlation heatmap, and distributions split by outcome
* Two ML pages:

  * **ML Model 1:** Logistic Regression with feature scaling
  * **ML Model 2:** Decision Tree (no scaling required)
* In-app training, evaluation (accuracy, confusion matrix, classification report) and simple prediction forms

---

## 📊Dataset

The app uses the Pima Indians Diabetes dataset (commonly used in ML examples). The dataset file is included as `diabetes.csv` in the repository root. It contains 768 rows and the following features:

* Pregnancies
* Glucose
* BloodPressure (renamed in app as `Diastolic Blood Pressure`)
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction (renamed in app as `Diabetes Pedigree Function`)
* Age
* Outcome (0 = non-diabetic, 1 = diabetic)

**Note on data values:** In this dataset, zero values in some columns (e.g., Glucose, BMI, BloodPressure, SkinThickness, Insulin) are treated as missing values rather than true zeros. The app replaces zeros with `NaN` for those fields and drops rows that remain with missing values — see the preprocessing section for details.

---

## 🧠How the app preprocesses data

The cleaning steps performed in `app.py` are reproduced here for clarity (see the code comments in `diabetes.py`):

1. Read `diabetes.csv` into a DataFrame.
2. Temporarily drop `Outcome`, `Pregnancies`, `Insulin`, `SkinThickness` into `df_temp` and replace zeros with `NaN` in that temporary frame. This step intends to mark zeros as missing for certain continuous fields.
3. Reconstruct the DataFrame by concatenating the original columns back in a specific order and then dropping rows containing any `NaN` values (so the app only uses fully complete rows).
4. Rename columns for readability: `BloodPressure` → `Diastolic Blood Pressure`, `DiabetesPedigreeFunction` → `Diabetes Pedigree Function`.

**Effect:** Rows with missing values (originally zero) are removed. This is a simple approach and reduces dataset size — more sophisticated imputation strategies (mean/median/knn) are alternatives.

---

## 🛠️Quick start - run locally

1. Clone the repository:

```bash
git clone https://github.com/shubhamawchare/Diabetes-Prediction-Web-App.git
cd Diabetes-Prediction-Web-App
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

*If you don't have `requirements.txt` yet, install the main packages used in the project:*

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
```

4. Ensure `diabetes.csv` is in the repository root (same directory as `diabetes.py`).

5. Run the app:

```bash
streamlit run app.py
```

6. A browser window should open (typically at `http://localhost:8501`). Use the sidebar to navigate.

---

## 🚀App pages & usage

**Home**

* Shows a title, short dataset description, and two dataframes: a preview (`head()`) and `describe()` statistics.

**Data Visualizations**

* Buttons to render pre-built plots: Age histogram, Glucose vs BMI scatterplot, and a correlation heatmap.
* Also offers distribution plots for every numeric feature split by `Outcome` (Diabetic vs Non-diabetic) using kernel density estimates.

**ML Model 1 — Logistic Regression**

* Choose features and test set size via controls in the sidebar.
* Training performs scaling (`StandardScaler`) before fitting a `LogisticRegression` model.
* Shows accuracy, confusion matrix, and classification report.
* After training the model is stored in Streamlit `session_state` and a prediction form appears.

**ML Model 2 — Decision Tree**

* Similar feature selection and test size controls.
* Trains a `DecisionTreeClassifier` and displays performance metrics.
* After training the model is stored in session state and a prediction form appears.

---

## Model details & implementation notes

* **Logistic Regression:** The app scales features with `StandardScaler` prior to training and prediction (important for models that assume standardized input).
* **Decision Tree:** Trained on raw features (no scaling). Decision trees are sensitive to overfitting — pruning or setting `max_depth`/`min_samples_leaf` is recommended for production use.
* **Saving models:** Currently the app stores trained models in `st.session_state` only (live session). If you want to persist models between runs, add `joblib.dump()` or `pickle` to save the model to disk and `joblib.load()` to reload it.

---

## Known limitations & notes

* This app performs **row-wise dropping** of any rows that contain zero values in the columns that were considered missing. This can reduce the available data and potentially bias results.
* The dataset is specific (Pima Indian women) — results and models are **not** generalizable to broader populations without further validation.
* No hyperparameter tuning or cross-validation is implemented. Model evaluation is based on a single split controlled by `random_state=0`.
* Not a medical product — do not use the predictions for clinical decisions.

---

## 🔮Suggestions & next steps (ideas for improvement)

* Implement imputation strategies (median/mean/KNN) instead of dropping rows.
* Add cross-validation and hyperparameter search (`GridSearchCV` / `RandomizedSearchCV`).
* Persist trained models to disk and add a UI to load previously saved models.
* Add user-friendly input validation, ranges, and helpful defaults on the prediction forms.
* Add unit tests for preprocessing and core functions.
* Replace `seaborn`/`matplotlib` calls with Plotly for interactive plots inside Streamlit.

---

## 📦Repository structure 

```
/ (repo root)
├─ diabetes.py
├─ diabetes.csv
├─ README.md
├─ requirements.txt
```

---

## 📝License

This project is provided under the **MIT License** by default.

---
