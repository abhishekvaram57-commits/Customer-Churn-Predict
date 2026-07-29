import joblib
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_explainer():
    return joblib.load("explainer.pkl")

explainer = load_explainer()

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Customer Churn Prediction")
st.sidebar.header("Customer Information")
st.markdown("""
This app is used to predict whether a telecom customer is likely to churn using an **XGBoost** model which had an test accuracy of ~ 81%.

The prediction is further accompanied by **SHAP explainability**, which shows the features the affects the model's final decision
""")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)
gender = 1 if gender == "Male" else 0


SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)
SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0


Partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)
Partner = 1 if Partner == "Yes" else 0


Dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)
Dependents = 1 if Dependents == "Yes" else 0


tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    step=1
)


PhoneService = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)
PhoneService = 1 if PhoneService == "Yes" else 0


PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0


MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0
)


TotalCharges = tenure * MonthlyCharges

st.sidebar.write(
    f"Estimated Total Charges: ${TotalCharges:.2f}"
)

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0
MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
InternetService_No = 1 if InternetService == "No" else 0

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineSecurity_No_internet_service = 1 if OnlineSecurity == "No internet service" else 0
OnlineSecurity_Yes = 1 if OnlineSecurity == "Yes" else 0

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

OnlineBackup_No_internet_service = 1 if OnlineBackup == "No internet service" else 0
OnlineBackup_Yes = 1 if OnlineBackup == "Yes" else 0

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

DeviceProtection_No_internet_service = 1 if DeviceProtection == "No internet service" else 0
DeviceProtection_Yes = 1 if DeviceProtection == "Yes" else 0


TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

TechSupport_No_internet_service = 1 if TechSupport == "No internet service" else 0
TechSupport_Yes = 1 if TechSupport == "Yes" else 0

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingTV_No_internet_service = 1 if StreamingTV == "No internet service" else 0
StreamingTV_Yes = 1 if StreamingTV == "Yes" else 0

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

StreamingMovies_No_internet_service = 1 if StreamingMovies == "No internet service" else 0
StreamingMovies_Yes = 1 if StreamingMovies == "Yes" else 0

Contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

Contract_One_year = 1 if Contract == "One year" else 0
Contract_Two_year = 1 if Contract == "Two year" else 0

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

PaymentMethod_Credit_card_automatic = 1 if PaymentMethod == "Credit card (automatic)" else 0
PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0


customer = [[
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    PaperlessBilling,
    MonthlyCharges,
    TotalCharges,
    MultipleLines_No_phone_service,
    MultipleLines_Yes,
    InternetService_Fiber_optic,
    InternetService_No,
    OnlineSecurity_No_internet_service,
    OnlineSecurity_Yes,
    OnlineBackup_No_internet_service,
    OnlineBackup_Yes,
    DeviceProtection_No_internet_service,
    DeviceProtection_Yes,
    TechSupport_No_internet_service,
    TechSupport_Yes,
    StreamingTV_No_internet_service,
    StreamingTV_Yes,
    StreamingMovies_No_internet_service,
    StreamingMovies_Yes,
    Contract_One_year,
    Contract_Two_year,
    PaymentMethod_Credit_card_automatic,
    PaymentMethod_Electronic_check,
    PaymentMethod_Mailed_check
]]

columns = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'PaperlessBilling',
    'MonthlyCharges',
    'TotalCharges',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_No internet service',
    'OnlineSecurity_Yes',
    'OnlineBackup_No internet service',
    'OnlineBackup_Yes',
    'DeviceProtection_No internet service',
    'DeviceProtection_Yes',
    'TechSupport_No internet service',
    'TechSupport_Yes',
    'StreamingTV_No internet service',
    'StreamingTV_Yes',
    'StreamingMovies_No internet service',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

customer_df = pd.DataFrame(customer, columns=columns)

if st.button("Predict Churn"):

    prediction = model.predict(customer_df)
    probability = model.predict_proba(customer_df)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Stay Probability",
            f"{probability[0][0]*100:.2f}%"
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability[0][1]*100:.2f}%"
        )

    st.subheader("Churn Risk")
    st.progress(float(probability[0][1]))

    st.subheader("Customer Details")
    st.dataframe(customer_df)
    shap_values = explainer.shap_values(customer_df)
    st.subheader("Why did the model make this prediction?")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[0],
    customer_df.iloc[0],
    feature_names=customer_df.columns,
    show=False
    )
    st.pyplot(fig)
    plt.close(fig)

      

