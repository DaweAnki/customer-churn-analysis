# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv('../data/cleaned/telco_churn_dashboard.csv')

@st.cache_resource
def load_model():
    model = joblib.load('../src/models/churn_model.pkl')
    scaler = joblib.load('../src/models/scaler.pkl')
    return model, scaler

df = load_data()
model, scaler = load_model()
# Sidebar
st.sidebar.markdown("""
    <div style='
        background: linear-gradient(180deg, #1B3A6B 0%, #0D1F3C 100%);
        padding: 12px 8px 8px 8px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 5px;
    '>
        <div style='font-size: 30px;'>📡</div>
        <div style='color: white; font-size: 14px; font-weight: bold; margin: 3px 0px 1px 0px;'>Customer Churn</div>
        <div style='color: #4A90D9; font-size: 9px;'>Telecom Analytics | 🐍 Python | 📊 PowerBI | 🤖 ML</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p style='margin: 3px 0px; font-size: 13px;'>🔍 <b>Filters</b></p>", unsafe_allow_html=True)

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

churn_filter = st.sidebar.multiselect(
    "Churn Status",
    options=df['Churn'].unique(),
    default=df['Churn'].unique()
)

# Apply Filters
df_filtered = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter)) &
    (df['Churn'].isin(churn_filter))
]

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Ankita Daweshar**")
st.sidebar.markdown("[GitHub](https://github.com/DaweAnki) | [LinkedIn](https://www.linkedin.com/in/ankita-daweshar-4a820b318/)")
# Main Title
st.title("📊 Customer Churn Analysis Dashboard")
st.markdown("Telecom customer churn analysis using ML | Dataset: Kaggle Telco Churn")
st.markdown("---")

# Create 3 Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🔍 EDA", "🤖 Predict Churn"])

# ── TAB 1: OVERVIEW ──
with tab1:
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df_filtered):,}")

    with col2:
        churn_rate = (df_filtered['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        avg_charges = df_filtered['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

    with col4:
        avg_tenure = df_filtered['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
# Overview Charts
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Churn Distribution Pie Chart
        churn_counts = df_filtered['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            color_discrete_sequence=['#1B3A6B', '#9DC3E6'],
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by Contract Type
        contract_churn = df_filtered.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate %']
        fig = px.bar(
            contract_churn,
            x='Churn Rate %',
            y='Contract',
            orientation='h',
            title="Churn Rate by Contract Type",
            color_discrete_sequence=['#1B3A6B']
        )
        st.plotly_chart(fig, use_container_width=True)
# ── TAB 2: EDA ──
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Tenure Distribution
        fig = px.histogram(
            df_filtered,
            x='tenure',
            color='Churn',
            title="Tenure Distribution by Churn",
            color_discrete_sequence=['#1B3A6B', '#9DC3E6'],
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Monthly Charges Box Plot
        fig = px.box(
            df_filtered,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges by Churn",
            color='Churn',
            color_discrete_sequence=['#1B3A6B', '#9DC3E6']
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    
    with col3:
        # Churn by Internet Service
        internet_churn = df_filtered.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        internet_churn.columns = ['InternetService', 'Churn Rate %']
        fig = px.bar(
            internet_churn,
            x='InternetService',
            y='Churn Rate %',
            title="Churn Rate by Internet Service",
            color_discrete_sequence=['#1B3A6B']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Churn by Payment Method
        payment_churn = df_filtered.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        payment_churn.columns = ['PaymentMethod', 'Churn Rate %']
        fig = px.bar(
            payment_churn,
            x='Churn Rate %',
            y='PaymentMethod',
            orientation='h',
            title="Churn Rate by Payment Method",
            color_discrete_sequence=['#2E75B6']
        )
        st.plotly_chart(fig, use_container_width=True)
# ── TAB 3: PREDICT CHURN ──
with tab3:
    st.subheader("🤖 Predict Customer Churn")
    st.markdown("Fill in the customer details below to predict churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
# Predict Button
    st.markdown("---")
    if st.button("🔮 Predict Churn", use_container_width=True):

        # Build input data
        input_data = {
            'gender': 0, 'SeniorCitizen': senior_citizen,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure, 'PhoneService': 1,
            'MultipleLines': 0,
            'OnlineSecurity': 1 if online_security == 'Yes' else 0,
            'OnlineBackup': 0, 'DeviceProtection': 0,
            'TechSupport': 1 if tech_support == 'Yes' else 0,
            'StreamingTV': 0, 'StreamingMovies': 0,
            'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn_num': 0,
            'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
            'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No': 1 if internet_service == 'No' else 0,
            'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
            'Contract_One year': 1 if contract == 'One year' else 0,
            'Contract_Two year': 1 if contract == 'Two year' else 0,
            'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
            'charge_ratio': monthly_charges / (total_charges + 1),
            'total_services': sum([online_security == 'Yes', tech_support == 'Yes'])
        }

        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # Predict
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # Show Result
        st.markdown("### Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Churn Probability", f"{probability:.1f}%")

        with col2:
            if probability >= 70:
                st.error("🔴 HIGH RISK — This customer is likely to churn!")
            elif probability >= 40:
                st.warning("🟡 MEDIUM RISK — Monitor this customer closely.")
            else:
                st.success("🟢 LOW RISK — This customer is likely to stay!")

        # Progress bar
        st.progress(int(probability))
        st.caption(f"Model confidence: {probability:.1f}% probability of churn")