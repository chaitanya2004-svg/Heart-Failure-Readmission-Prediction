import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import calendar
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration
st.set_page_config(
    page_title="Heart Failure Readmission Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* New CSS for enhanced features */
    .health-tip {
        background-color: #1a237e;  /* Deep blue background */
        color: #ffffff;  /* White text */
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #4caf50;  /* Green accent border */
        font-size: 1.1em;
        line-height: 1.5;
    }
    .achievement-badge {
        text-align: center;
        padding: 10px;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        margin: 10px;
        display: inline-block;
    }
    .mood-tracker {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("❤️ Heart Failure Readmission Prediction")
st.markdown("""
    Welcome to the Heart Failure Readmission Prediction! This interactive tool helps you explore and understand
    heart failure data through various visualizations and insights using traditional machine learning approaches.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Analysis", "Predictions", "Emergency Measures", "About the Dashboard"])

# Sample data generation with more realistic features
def generate_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate realistic medical data
    data = {
        'Date': dates,
        'Age': np.random.normal(65, 10, n_samples).clip(40, 90),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Heart Rate': np.random.normal(75, 10, n_samples).clip(40, 120),
        'Blood Pressure': np.random.normal(120, 15, n_samples).clip(90, 160),
        'Oxygen Level': np.random.normal(98, 2, n_samples).clip(90, 100),
        'BMI': np.random.normal(28, 5, n_samples).clip(18, 40),
        'Diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Smoking': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Exercise': np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], n_samples),
        'Cholesterol': np.random.normal(200, 30, n_samples).clip(150, 300),
        'Creatinine': np.random.normal(1.1, 0.3, n_samples).clip(0.5, 2.0)
    }
    
    # Calculate risk score using traditional ML features
    X = pd.DataFrame({
        'Age': data['Age'],
        'Heart Rate': data['Heart Rate'],
        'Blood Pressure': data['Blood Pressure'],
        'BMI': data['BMI'],
        'Diabetes': data['Diabetes'],
        'Smoking': data['Smoking'],
        'Cholesterol': data['Cholesterol'],
        'Creatinine': data['Creatinine']
    })
    
    # Simple risk calculation based on medical guidelines
    risk_score = (
        (data['Age'] - 40) / 50 * 0.2 +  # Age factor
        (data['Heart Rate'] - 60) / 60 * 0.1 +  # Heart rate factor
        (data['Blood Pressure'] - 90) / 70 * 0.15 +  # Blood pressure factor
        (data['BMI'] - 18) / 22 * 0.1 +  # BMI factor
        data['Diabetes'] * 0.2 +  # Diabetes factor
        data['Smoking'] * 0.15 +  # Smoking factor
        (data['Cholesterol'] - 150) / 150 * 0.1 +  # Cholesterol factor
        (data['Creatinine'] - 0.5) / 1.5 * 0.1  # Creatinine factor
    ).clip(0, 1)
    
    data['Risk Score'] = risk_score
    return pd.DataFrame(data)

def train_model():
    X = df[['Age', 'Heart Rate', 'Blood Pressure', 'BMI', 'Diabetes', 
            'Smoking', 'Cholesterol', 'Creatinine']]
    y = (df['Risk Score'] > 0.5).astype(int)  # Binary classification
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def get_pattern_description(risk_level):
    if risk_level < 0.3:
        return "Low risk pattern with minimal health concerns. Regular monitoring recommended."
    elif risk_level < 0.6:
        return "Moderate risk pattern. Some health factors need attention. Lifestyle modifications may be beneficial."
    else:
        return "High risk pattern. Close monitoring and medical intervention recommended. Multiple risk factors present."

# Initialize data and model
df = generate_sample_data()
model, X_test, y_test = train_model()

if page == "Overview":
    # Overview Page
    st.header("📊 Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Average Heart Rate", value=f"{df['Heart Rate'].mean():.1f} bpm")
    with col2:
        st.metric(label="Average BP", value=f"{df['Blood Pressure'].mean():.1f} mmHg")
    with col3:
        st.metric(label="Average O2 Level", value=f"{df['Oxygen Level'].mean():.1f}%")
    with col4:
        st.metric(label="Risk Level", value="Medium", delta="2%")

    # Interactive Chart
    st.subheader("📈 Heart Rate Trends")
    fig = px.line(df, x='Date', y='Heart Rate', 
                  title='Heart Rate Over Time',
                  template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    # Tabs for different analysis types
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Basic Analysis", "Advanced Insights", "Patient Comparison"
    ])
    
    with analysis_tab1:
        st.subheader("Basic Health Metrics Analysis")
        
        # Interactive Filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Select Date Range", 
                                     [df['Date'].min(), df['Date'].max()])
            metric = st.selectbox("Select Primary Metric", 
                                ['Heart Rate', 'Blood Pressure', 'Oxygen Level', 'BMI'])
        with col2:
            group_by = st.selectbox("Group By", 
                                  ['Day', 'Week', 'Month'])
        
        # Time Series Analysis
        st.subheader("📈 Time Series Analysis")
        # Select only numeric columns for resampling
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()
        df_numeric['Date'] = df['Date']
        
        if group_by == 'Day':
            grouped_data = df_numeric.set_index('Date').resample('D').mean()
        elif group_by == 'Week':
            grouped_data = df_numeric.set_index('Date').resample('W').mean()
        else:
            grouped_data = df_numeric.set_index('Date').resample('M').mean()
        
        # Update metric options to only include numeric columns
        metric = st.selectbox("Select Primary Metric", 
                            [col for col in numeric_columns if col != 'Date'])
        
        fig = px.line(grouped_data, y=metric,
                     title=f'{metric} Over Time',
                     template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{df[metric].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[metric].median():.2f}")
        with col3:
            st.metric("Standard Deviation", f"{df[metric].std():.2f}")
        
        # Distribution Plot
        st.subheader("📊 Distribution Analysis")
        fig = px.histogram(df, x=metric, nbins=30,
                          title=f'Distribution of {metric}',
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with analysis_tab2:
        st.subheader("Advanced Health Insights")
        
        # Correlation Analysis
        st.markdown("### 🔄 Correlation Analysis")
        corr_matrix = df[['Heart Rate', 'Blood Pressure', 'Oxygen Level', 'BMI', 
                         'Cholesterol', 'Creatinine', 'Risk Score']].corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix of Health Metrics')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Factor Analysis
        st.markdown("### ⚠️ Risk Factor Analysis")
        risk_factors = ['Age', 'BMI', 'Diabetes', 'Smoking', 'Cholesterol']
        risk_data = df[risk_factors + ['Risk Score']].copy()
        
        # Create risk factor impact visualization
        impact_data = []
        for factor in risk_factors:
            if factor in ['Diabetes', 'Smoking']:
                impact = risk_data.groupby(factor)['Risk Score'].mean().diff().iloc[-1]
            else:
                impact = risk_data[factor].corr(risk_data['Risk Score'])
            impact_data.append({'Factor': factor, 'Impact': impact})
        
        impact_df = pd.DataFrame(impact_data)
        fig = px.bar(impact_df, x='Factor', y='Impact',
                    title='Impact of Risk Factors on Heart Failure Risk',
                    color='Impact',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Detection
        st.markdown("### 🔍 Anomaly Detection")
        
        # Calculate anomalies using fixed threshold (2 standard deviations)
        threshold = 2.0  # Fixed threshold of 2 standard deviations
        
        for metric in ['Heart Rate', 'Blood Pressure', 'Oxygen Level']:
            mean = df[metric].mean()
            std = df[metric].std()
            anomalies = df[abs(df[metric] - mean) > threshold * std]
            
            # Display average value range
            st.markdown(f"""
            #### {metric} Analysis
            - **Average Value**: {mean:.2f}
            - **Standard Deviation**: {std:.2f}
            - **Normal Range**: {mean - threshold*std:.2f} to {mean + threshold*std:.2f}
            - **Threshold**: {threshold} standard deviations
            """)
            
            if not anomalies.empty:
                st.warning(f"Found {len(anomalies)} anomalies in {metric}")
                st.dataframe(anomalies[['Date', metric]].head())
                
                # Add visualization for anomalies
                fig = go.Figure()
                # Add normal data points
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df[metric],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=8)
                ))
                # Add anomaly points
                fig.add_trace(go.Scatter(
                    x=anomalies['Date'],
                    y=anomalies[metric],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=10, symbol='x')
                ))
                # Add threshold lines
                fig.add_hline(y=mean + threshold*std, line_dash="dash", line_color="red",
                            annotation_text="Upper Threshold", annotation_position="top right")
                fig.add_hline(y=mean - threshold*std, line_dash="dash", line_color="red",
                            annotation_text="Lower Threshold", annotation_position="bottom right")
                fig.add_hline(y=mean, line_dash="dot", line_color="green",
                            annotation_text="Mean", annotation_position="top left")
                
                fig.update_layout(
                    title=f'{metric} Anomalies Detection',
                    xaxis_title='Date',
                    yaxis_title=metric,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

    with analysis_tab3:
        st.subheader("Patient Comparison")
        
        # Patient Selection
        col1, col2 = st.columns(2)
        with col1:
            patient1 = st.selectbox("Select Patient 1", 
                                  df['Gender'].unique())
        with col2:
            patient2 = st.selectbox("Select Patient 2", 
                                  df['Gender'].unique())
        
        # Comparison Metrics
        metrics = ['Heart Rate', 'Blood Pressure', 'Oxygen Level', 'BMI', 'Risk Score']
        comparison_data = pd.DataFrame({
            'Metric': metrics,
            'Patient 1': [df[df['Gender'] == patient1][m].mean() for m in metrics],
            'Patient 2': [df[df['Gender'] == patient2][m].mean() for m in metrics]
        })
        
        # Radar Chart
        fig = go.Figure()
        for patient in ['Patient 1', 'Patient 2']:
            fig.add_trace(go.Scatterpolar(
                r=comparison_data[patient],
                theta=metrics,
                fill='toself',
                name=patient
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Comparison
        st.subheader("Detailed Comparison")
        st.dataframe(comparison_data)

elif page == "Predictions":
    st.header("🎯 Risk Predictions")
    
    # Tabs for different prediction modes
    pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5 = st.tabs([
        "Quick Assessment", "Detailed Analysis", "Historical Trends", 
        "Health Journal", "Lifestyle Guide"
    ])
    
    with pred_tab1:
        st.subheader("Quick Health Assessment")
        with st.form("quick_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 0, 100, 50)
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
                weight = st.number_input("Weight (kg)", 30, 200, 70)
                height = st.number_input("Height (cm)", 100, 250, 170)
            
            with col2:
                blood_pressure = st.number_input("Blood Pressure (mmHg)", 60, 200, 120)
                diabetes = st.checkbox("Diabetes")
                smoking = st.checkbox("Smoking History")
                cholesterol = st.number_input("Cholesterol (mg/dL)", 150, 300, 200)
            
            submitted = st.form_submit_button("Calculate Risk")
            
            if submitted:
                # Calculate BMI
                height_m = height / 100
                bmi = weight / (height_m * height_m)
                
                # Prepare input for model
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Heart Rate': [heart_rate],
                    'Blood Pressure': [blood_pressure],
                    'BMI': [bmi],
                    'Diabetes': [1 if diabetes else 0],
                    'Smoking': [1 if smoking else 0],
                    'Cholesterol': [cholesterol],
                    'Creatinine': [1.1]  # Default value
                })
                
                # Get prediction
                risk_prob = model.predict_proba(input_data)[0][1]
                
                # Display results
                st.success(f"Predicted Risk Score: {risk_prob:.2%}")
                
                # Risk visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_prob * 100,
                    title={'text': "Risk Level"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "darkblue"},
                          'steps': [
                              {'range': [0, 30], 'color': "lightgreen"},
                              {'range': [30, 70], 'color': "yellow"},
                              {'range': [70, 100], 'color': "red"}
                          ]}))
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance metrics
                st.subheader("Model Performance")
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                # Feature importance
                st.subheader("Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Feature', y='Importance',
                            title='Feature Importance in Risk Prediction')
                st.plotly_chart(fig, use_container_width=True)

    with pred_tab2:
        st.subheader("Detailed Health Analysis")
        st.markdown("""
        ### Comprehensive Health Assessment
        This section provides a detailed analysis of your health metrics and their impact on heart failure risk.
        """)
        
        # Advanced metrics
        col1, col2 = st.columns(2)
        with col1:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50)
            ldl = st.number_input("LDL Cholesterol (mg/dL)", 50, 200, 100)
        
        with col2:
            triglycerides = st.number_input("Triglycerides (mg/dL)", 50, 500, 150)
            glucose = st.number_input("Fasting Glucose (mg/dL)", 70, 200, 100)
            creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 2.0, 1.0)
        
        if st.button("Analyze Health Metrics"):
            # Calculate additional risk factors
            lipid_ratio = cholesterol / hdl
            metabolic_risk = (glucose > 100) + (triglycerides > 150) + (hdl < 40)
            
            # Create detailed analysis visualization
            metrics = {
                'Lipid Ratio': lipid_ratio,
                'Metabolic Risk Score': metabolic_risk,
                'Kidney Function': 1 if creatinine > 1.2 else 0
            }
            
            fig = px.bar(x=list(metrics.keys()), 
                        y=list(metrics.values()),
                        title="Health Metrics Analysis",
                        labels={'x': 'Metric', 'y': 'Score'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed recommendations
            st.subheader("Detailed Recommendations")
            if lipid_ratio > 4:
                st.warning("High lipid ratio detected. Consider dietary changes and exercise.")
            if metabolic_risk > 1:
                st.warning("Multiple metabolic risk factors present. Regular monitoring recommended.")
            if creatinine > 1.2:
                st.warning("Kidney function may need attention. Consult healthcare provider.")

    with pred_tab3:
        st.subheader("Historical Risk Trends")
        st.markdown("""
        ### Track Your Progress
        Monitor your risk factors over time to see the impact of lifestyle changes.
        """)
        
        # Simulate historical data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        historical_data = pd.DataFrame({
            'Date': dates,
            'Risk Score': np.random.normal(0.5, 0.1, len(dates)).clip(0, 1),
            'Heart Rate': np.random.normal(75, 5, len(dates)),
            'Blood Pressure': np.random.normal(120, 10, len(dates))
        })
        
        # Plot historical trends
        fig = px.line(historical_data, x='Date', y=['Risk Score', 'Heart Rate', 'Blood Pressure'],
                     title='Health Metrics Over Time',
                     labels={'value': 'Measurement', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Progress summary
        st.subheader("Progress Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score Change", 
                     f"{(historical_data['Risk Score'].iloc[-1] - historical_data['Risk Score'].iloc[0])*100:.1f}%")
        with col2:
            st.metric("Heart Rate Change", 
                     f"{historical_data['Heart Rate'].iloc[-1] - historical_data['Heart Rate'].iloc[0]:.1f} bpm")
        with col3:
            st.metric("BP Change", 
                     f"{historical_data['Blood Pressure'].iloc[-1] - historical_data['Blood Pressure'].iloc[0]:.1f} mmHg")

    with pred_tab4:
        st.subheader("📝 Health Journal")
        
        # Journal Entry
        with st.expander("New Journal Entry", expanded=True):
            date = st.date_input("Date", datetime.now())
            mood = st.select_slider("How are you feeling today?", 
                                  options=["😢", "😕", "😐", "🙂", "😊"])
            
            col1, col2 = st.columns(2)
            with col1:
                sleep_hours = st.number_input("Hours of Sleep", 0, 24, 8)
                water_intake = st.number_input("Glasses of Water", 0, 20, 8)
            
            with col2:
                exercise_minutes = st.number_input("Exercise Minutes", 0, 300, 30)
                stress_level = st.select_slider("Stress Level", 
                                             options=["Low", "Moderate", "High", "Very High"])
            
            notes = st.text_area("Notes", "How was your day? Any symptoms or concerns?")
            
            if st.button("Save Entry"):
                st.success("Journal entry saved!")
                
                # Display mood trend
                st.subheader("Mood Trends")
                mood_data = pd.DataFrame({
                    'Date': [date],
                    'Mood': [mood],
                    'Sleep': [sleep_hours],
                    'Exercise': [exercise_minutes]
                })
                fig = px.line(mood_data, x='Date', y=['Sleep', 'Exercise'],
                             title='Daily Activity Tracking')
                st.plotly_chart(fig, use_container_width=True)
        
        # Health Tips
        st.subheader("💡 Daily Health Tip")
        health_tips = [
            "Stay hydrated! Aim for 8 glasses of water daily.",
            "Take a 5-minute walk every hour to improve circulation.",
            "Practice deep breathing exercises to reduce stress.",
            "Include more fruits and vegetables in your diet.",
            "Get 7-8 hours of sleep for optimal heart health."
        ]
        st.markdown(f'<div class="health-tip">{random.choice(health_tips)}</div>', 
                   unsafe_allow_html=True)
        
        # Achievements
        st.subheader("🏆 Achievements")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="achievement-badge" style="background-color: #ffd700;">🌟<br>7 Day Streak</div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="achievement-badge" style="background-color: #90caf9;">💧<br>Hydration Master</div>', 
                       unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="achievement-badge" style="background-color: #a5d6a7;">🏃<br>Exercise Pro</div>', 
                       unsafe_allow_html=True)

    with pred_tab5:
        st.subheader("🌱 Lifestyle Guide")
        
        # Interactive Lifestyle Assessment
        st.markdown("### Your Lifestyle Assessment")
        
        # Diet Section
        with st.expander("🍽️ Diet & Nutrition", expanded=True):
            st.markdown("#### Daily Food Intake")
            col1, col2 = st.columns(2)
            with col1:
                fruits = st.slider("Servings of Fruits", 0, 10, 2)
                vegetables = st.slider("Servings of Vegetables", 0, 10, 3)
            with col2:
                protein = st.slider("Servings of Lean Protein", 0, 10, 2)
                whole_grains = st.slider("Servings of Whole Grains", 0, 10, 3)
            
            # Diet Score
            diet_score = (fruits + vegetables + protein + whole_grains) / 40 * 100
            st.progress(diet_score / 100)
            st.metric("Diet Score", f"{diet_score:.1f}%")
            
            # Diet Recommendations
            if diet_score < 60:
                st.warning("Consider increasing your intake of fruits, vegetables, and whole grains.")
            elif diet_score < 80:
                st.info("Good diet! Try to include more variety in your meals.")
            else:
                st.success("Excellent diet! Keep up the good work!")
        
        # Exercise Section
        with st.expander("🏃 Exercise & Activity", expanded=True):
            st.markdown("#### Weekly Exercise Plan")
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            exercise_plan = {}
            
            for day in days:
                col1, col2 = st.columns([2, 1])
                with col1:
                    exercise_plan[day] = st.selectbox(
                        f"Activity for {day}",
                        ["Rest", "Walking", "Jogging", "Cycling", "Swimming", "Yoga", "Strength Training"]
                    )
                with col2:
                    if exercise_plan[day] != "Rest":
                        exercise_plan[f"{day}_duration"] = st.number_input(
                            "Minutes",
                            min_value=10,
                            max_value=180,
                            value=30,
                            key=f"duration_{day}"
                        )
            
            # Exercise Summary
            st.subheader("Weekly Exercise Summary")
            exercise_data = pd.DataFrame({
                'Day': days,
                'Activity': [exercise_plan[day] for day in days],
                'Duration': [exercise_plan.get(f"{day}_duration", 0) for day in days]
            })
            fig = px.bar(exercise_data, x='Day', y='Duration',
                        color='Activity',
                        title='Weekly Exercise Plan')
            st.plotly_chart(fig, use_container_width=True)
        
        # Stress Management
        with st.expander("🧘 Stress Management", expanded=True):
            st.markdown("#### Stress Management Techniques")
            techniques = {
                "Deep Breathing": "Practice 4-7-8 breathing technique",
                "Meditation": "Start with 5 minutes daily meditation",
                "Progressive Muscle Relaxation": "Systematically tense and relax muscle groups",
                "Mindful Walking": "Take a 10-minute mindful walk in nature",
                "Journaling": "Write down your thoughts and feelings"
            }
            
            selected_technique = st.selectbox(
                "Choose a technique to learn",
                list(techniques.keys())
            )
            
            st.markdown(f"### {selected_technique}")
            st.info(techniques[selected_technique])
            
            # Timer for practice
            if st.button("Start Practice Timer"):
                st.markdown("⏱️ 5-minute practice timer started")
                st.progress(0)

elif page == "Emergency Measures":
    st.header("🚨 Emergency Measures")
    
    # Emergency Symptoms
    st.markdown("""
    ### ⚠️ Emergency Symptoms
    Seek immediate medical attention if you experience:
    - Severe chest pain or pressure
    - Difficulty breathing or shortness of breath
    - Fainting or severe dizziness
    - Rapid or irregular heartbeat
    - Severe swelling in legs or ankles
    - Sudden confusion or difficulty speaking
    - Blue tint to lips or fingers
    """)
    
    # Emergency Response Steps
    st.markdown("""
    ### 🚑 Emergency Response Steps
    
    1. **Stay Calm**
       - Take deep breaths
       - Sit or lie down in a comfortable position
       - Call for help if needed
    
    2. **Call Emergency Services (911)**
       - Clearly state your location
       - Describe your symptoms
       - Follow operator instructions
    
    3. **While Waiting for Help**
       - Loosen tight clothing
       - Take prescribed emergency medications if available
       - Keep emergency contact information readily available
       - Stay in a safe position
    """)
    
    # Emergency Kit Checklist
    st.markdown("""
    ### 🎒 Emergency Kit Checklist
    
    Essential items to keep readily available:
    - List of current medications
    - Medical history documents
    - Insurance information
    - Emergency contact numbers
    - Prescribed emergency medications
    - Basic first aid supplies
    - Water and snacks
    - Flashlight and batteries
    - Warm blanket
    """)
    
    # Emergency Medication Guide
    st.markdown("""
    ### 💊 Emergency Medication Guide
    
    **Common Emergency Medications:**
    - Nitroglycerin (for chest pain)
    - Aspirin (for heart attack symptoms)
    - Emergency inhalers (for breathing difficulties)
    
    **Important Notes:**
    - Keep medications in original containers
    - Check expiration dates regularly
    - Know proper dosage and administration
    - Store in a cool, dry place
    """)
    
    # Interactive Emergency Response Guide
    st.markdown("### 🎯 Interactive Emergency Response Guide")
    
    emergency_scenario = st.selectbox(
        "Select Emergency Scenario",
        ["Chest Pain", "Difficulty Breathing", "Fainting", "Rapid Heartbeat", "Severe Swelling"]
    )
    
    if emergency_scenario == "Chest Pain":
        st.markdown("""
        **Immediate Actions:**
        1. Stop all activity and rest
        2. Take prescribed nitroglycerin if available
        3. Call 911 if pain persists
        4. Chew one regular aspirin (unless allergic)
        5. Stay calm and wait for emergency services
        """)
    elif emergency_scenario == "Difficulty Breathing":
        st.markdown("""
        **Immediate Actions:**
        1. Sit upright
        2. Use prescribed inhaler if available
        3. Call 911 if breathing doesn't improve
        4. Loosen tight clothing
        5. Stay calm and focus on breathing
        """)
    elif emergency_scenario == "Fainting":
        st.markdown("""
        **Immediate Actions:**
        1. Lie down with legs elevated
        2. Call 911 if unconscious
        3. Check breathing and pulse
        4. Loosen tight clothing
        5. Stay with the person until help arrives
        """)
    elif emergency_scenario == "Rapid Heartbeat":
        st.markdown("""
        **Immediate Actions:**
        1. Sit or lie down
        2. Take deep breaths
        3. Call 911 if accompanied by chest pain
        4. Try vagal maneuvers if prescribed
        5. Monitor heart rate if possible
        """)
    elif emergency_scenario == "Severe Swelling":
        st.markdown("""
        **Immediate Actions:**
        1. Elevate affected area
        2. Call 911 if accompanied by chest pain
        3. Remove tight clothing or jewelry
        4. Apply cool compress if appropriate
        5. Monitor for breathing difficulties
        """)
    
    # Emergency Preparedness Tips
    st.markdown("""
    ### 💡 Emergency Preparedness Tips
    
    1. **Regular Check-ups**
       - Keep all medical appointments
       - Update emergency contacts
       - Review emergency plan with family
    
    2. **Documentation**
       - Keep medical records updated
       - Maintain current medication list
       - Have insurance information ready
    
    3. **Communication**
       - Share emergency plan with family
       - Know how to contact emergency services
       - Keep phone charged and accessible
    """)

else:  # About the Dashboard page
    st.header("ℹ️ About the Dashboard")
    st.markdown("""
    ### About This Dashboard
    
    This dashboard provides a comprehensive view of heart failure data and risk analysis.
    
    #### Features:
    - Real-time data visualization
    - Interactive risk prediction
    - Correlation analysis
    - Trend monitoring
    - Health journaling
    - Lifestyle guidance
    
    #### Data Sources:
    - Patient monitoring systems
    - Medical records
    - Real-time sensors
    
    #### Technology Stack:
    - Streamlit for the web interface
    - Plotly for interactive visualizations
    - Scikit-learn for machine learning
    - Pandas for data manipulation
    
    #### Contact:
    For more information, please contact the Hearteliligence team.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by the Team Heartelligence")
st.markdown("Team Members:")
st.markdown("Chaitanya (chaitanya.ghanghas@gmail.com)")
st.markdown("Yash Bagga (yashbagga5@gmail.com)")
st.markdown("Aditi (1.aditi.238@gmail.com)")