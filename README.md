# Heartelligence
❤️ **HEART FAILURE READMISSION PREDICTION** 


This project focuses on predicting hospital readmissions for heart failure patients and providing visual health insights through an interactive dashboard. It combines clinical data processing, machine learning modeling, and interactive health analytics using Python libraries and Streamlit.


🎯 Objectives
- Filter and extract heart failure cases from MIMIC-III.
- Predict 30-day readmission risk using machine learning.
- Visualize key patient metrics interactively.
- Empower patients with journaling and lifestyle tracking tools.
- Provide emergency response education for at-risk individuals.


📊 Features


- MIMIC-III dataset integration📁 
- Data preprocessing and heart failure diagnosis filtering⚙️ 
- Machine learning prediction of readmission risk 🤖 
- Health trends visualization📈
- Basic Health Matrices🔬
- Patient Comparison🥼
- Risk score calculator🧠 
- Correlation and anomaly detection📉 
- Health journaling👨‍⚕️
- Lifestyle tracking🧾 
- Emergency preparedness guidance🚨 
- Summary and performance reports📋 

🖼️ UI Preview

- Home Page
![Home Page](https://github.com/yashbagga5/Heartelligence/blob/c1622f530c440668295e4717d6f73cf2dd9dba93/Home%20page%201.jpg)
![Home Page Continued](https://github.com/yashbagga5/Heartelligence/blob/cb64642c703c145fdc19232c78c5c2a6beb3c045/Home%20Page%202.jpg)


🧪 Technologies Used

- Pandas / NumPy:	Data loading and preprocessing
- Scikit-learn:	Model training, evaluation, metrics
- XGBoost:	Predictive modeling (classification)
- Streamlit:	Web-based interactive UI
- Plotly:	Rich, interactive visualizations
- MIMIC-III Dataset:	Real-world medical data (ADMISSIONS.csv, DIAGNOSES_ICD.csv, PATIENTS.csv)


📚 Explanation:

- pandas as pd: Used for data manipulation and analysis (especially with tables called DataFrames).
- numpy as np: Used for numerical operations, especially with arrays.
- datetime, timedelta: Used for handling and manipulating dates and times.
- train_test_split: Function from sklearn to split your dataset into training and test sets.
- classification_report: Generates precision, recall, F1-score, and support metrics.
- XGBClassifier: A high-performance machine learning model (gradient boosting) from XGBoost, used for classification tasks.

**MACHINE LEARNING APPROACH** 


📁 Data Preparation (Notebook)

- Load CSVs from the MIMIC-III dataset
- Filter diagnosis codes for heart failure (ICD-9 codes)
- Merge relevant datasets: Diagnoses + Admissions + Patients
- Compute readmission risk based on features
- Train-test split and model evaluation

🤖 Model

- XGBoostClassifier in Jupyter Notebook
- RandomForestClassifier in Streamlit dashboard
- Evaluation metrics: accuracy, ROC-AUC, classification report

🌐 Dashboard Sections

- **Data Analysis:	Summary statistics, time series, anomaly detection**


![Data Analysis](https://github.com/yashbagga5/Heartelligence/blob/1a36e302f532b9381e3e1b51856b325a3939c6da/Data%20Analysis%20(Basic%20Analysis).jpg)
![Data Analysis](https://github.com/yashbagga5/Heartelligence/blob/1a36e302f532b9381e3e1b51856b325a3939c6da/Data%20Analysis%20(Advanced%20Insights).jpg)
![Data Analysis](https://github.com/yashbagga5/Heartelligence/blob/1a36e302f532b9381e3e1b51856b325a3939c6da/Data%20Analysis%20(patient%20Comparison).jpg)


- **Predictions:	Risk prediction form and model performance**


![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Quick%20Assement).jpg)
![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Quick%20Assement)%202.jpg)
![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Detailed%20Analysis)%20.jpg)
![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Hiatorical%20Trends)%20.jpg)
![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Health%20Journal).jpg)
![Predictions](https://github.com/yashbagga5/Heartelligence/blob/305302866d78d50b7edc7ef0ef306472c8cb2c19/Prediction%20(Lifstyle%20guide).jpg)


- **Emergency Measures:	Instructions and preparedness guide for emergencies**


![Emergency Measures](https://github.com/yashbagga5/Heartelligence/blob/ccb72e1b3aa49ab6ee5e311495da0a9cc7941c28/Emergency%20measures.jpg)
![Emergency Measures](https://github.com/yashbagga5/Heartelligence/blob/ccb72e1b3aa49ab6ee5e311495da0a9cc7941c28/Emergenncy%20guide%20(Quick%20Response%20guide).jpg)


- **About:	Tech stack and team information**


![About](https://github.com/yashbagga5/Heartelligence/blob/8fed342e39c5218037bacc3e5c3f6277013d6462/About%20the%20dashboard.jpg)
![About](https://github.com/yashbagga5/Heartelligence/blob/10b9e45860c0133f7a86585a075eecb699fc4ec8/About%20the%20dashboard%202.jpg)

🏗️ PROJECT ARCHITECTURE

**1. Backend (Data Processing & Modeling – Jupyter Notebook)**
   
  a) Data Sources: CSV files from the MIMIC-III clinical database:
  
    - ADMISSIONS.csv (admission data)
    - DIAGNOSES_ICD.csv (diagnosis codes)
    - PATIENTS.csv (demographics)

  b) Preprocessing Steps:
  
    - Filtered diagnosis codes related to heart failure (ICD-9).
    - Merged patient, diagnosis, and admission data.
    - Calculated time between discharges and next admissions.
    - Labeled readmissions within 30 days (readmit_30d).
    - Performed feature engineering (e.g., age, gender).

  c) Model Training:
  
    - Split the data into training and test sets.
    - Trained an XGBoostClassifier to predict 30-day readmission.
    - Evaluated the model using accuracy, precision, recall, F1 score, and AUC-ROC.
    - Saved feature importance plots and prediction functions.

**2. Frontend (Data Analysis & Interactive Dashboard)**
   
  a) Exploratory Data Analysis (Notebook)
  
    - Class Balance Check: Visualized how many patients were readmitted within 30 days.
    - Correlation Heatmaps: To identify relationships between features.    
    - Bar Plots and Histograms: For age distribution, gender breakdown, and length of stay.
    - Top Feature Visualizations: Feature importance plotted using sns.barplot.

   b) Interactive Data Exploration
   
     The dashboard offers rich interactivity powered by Plotly and Pandas:

     🔍 Overview Page
     
      - Key metrics: average heart rate, blood pressure, oxygen level.
      - Time-series trend charts for vitals.

      📊 Basic Analysis Tab
      
      Users can select:
      - Custom date ranges
      - Metrics like heart rate, BMI, etc.
      - Time grouping (daily, weekly, monthly)
      - Display summary statistics: mean, median, standard deviation.
      - Distribution plots with adjustable bins.

      🧠 Advanced Insights Tab
      
      - Correlation Matrix: Heatmap of inter-metric correlations.
      - Risk Factor Analysis: Measures how individual features impact risk score.
      - Anomaly Detection:
      - Uses z-score logic (2 standard deviations) to find outliers.
      - Visual markers highlight anomalies on time-series plots.

      🧑‍⚕️ Patient Comparison Tab
      
      Compares two patient groups (e.g., Male vs. Female) across:
      - Heart rate
      - Blood pressure
      - Oxygen level
      - BMI
      - Risk score
      - Radar chart and tabular comparisons.

     💡 Tools Used
     
      - pandas / numpy: Data manipulation and numerical analysis.
      - matplotlib / seaborn: Static plots and graphs in the notebook.
      - plotly.express / plotly.graph_objects: Interactive plots in Streamlit.
      - scikit-learn: For scaling, training, and evaluating models.
      - streamlit: For building real-time, interactive dashboards.

📂 Project Structure
      
      ├── Untitled1.ipynb                # Data analysis and model building in Jupyter Notebook
      ├── app.py                         # Streamlit dashboard code
      ├── README.md                      # Project documentation (this file)
      └── data/                          # Directory for MIMIC-III CSVs (e.g., ADMISSIONS.csv, DIAGNOSES_ICD.csv)


🛠️ FLOW CHART
      
      CSV Files (MIMIC-III) 
          ↓
      Data Cleaning & Feature Engineering (Notebook)
          ↓
      Model Training (XGBoost)
          ↓
      Model Evaluation (classification_report, AUC)
          ↓
      Streamlit UI with RandomForest on Simulated Data
          ↓
      Interactive Risk Prediction + Visualization + Health Tools


📈 Model Performance

      - Accuracy: ~86%
      - Precision: ~90%
      - f1 Score: ~87%
      - Confusion Matrix:   [10  1]
                            [ 2  9]
      - Key Features: Age, BMI, Heart Rate, Blood Pressure, Diabetes, Smoking, Cholesterol

**Accuracy**
![Accuracy](https://github.com/yashbagga5/Heartelligence/blob/ff7bb5f55ee72b9a4b85083be829f2d454eced0e/Accuracy.jpg)


**Confusion Matrix**
![Confusion Matrix](https://github.com/yashbagga5/Heartelligence/blob/0e5fd53d6f577ee3affa99a3e71264d4c13e34bc/Confusion%20matrix.jpg)


**ROC CURVE**

![ROC Curve](https://github.com/yashbagga5/Heartelligence/blob/541abe6bd16bdf5669312ba891bc3487e7f733b5/ROC%20Curve.jpg)

▶️ How to Run

1. Clone or Download the Project
    - git clone https://github.com/yashbagga5/Heartelligence
    - cd heartelligence
   
2. Install Requirements
   - pip install -r requirements.txt
   
3. Run the Dashboard
   - streamlit run app.py

**OVERVIEW VIDEO**
![Overview Video](https://github.com/yashbagga5/Heartelligence/blob/fd481112f68e4e79c9c9b8273947a77a0119cce6/Heart%20Failure%20Readmission%20Prediction%20Video%20overwiew.mp4)



🚀 Future Scope

As healthcare becomes increasingly data-driven, this project has strong potential to evolve into a real-world decision-support tool. Here are some ways this system can be extended and improved:

1. 🔬 Integration with Real-World Clinical Data
2. 🧠 Deep Learning Models
3. 📈 Time-Series Forecasting
4. 🧪 Explainable AI (XAI)
5. 📱 Mobile Health App
6. 🩺 Real-time Sensor Integration
7. 🏥 Clinician Dashboard
8. 📋 Automatic Emergency Alerts

🧑‍💻 Authors
1. Chaitanya — chaitanya.ghanghas@gmail.com
2. Yash Bagga — yashbagga5@gmail.com
3. Aditi — 1.aditi.238@gmail.com

