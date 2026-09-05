#Alerting options
#Print to console (above)
#Send email via SMTP
#Push to Slack or webhook
#Create tickets in monitoring system
#Step 5 Dashboard and Reporting
#Purpose Provide a quick Streamlit dashboard to inspect portfolio metrics, top exposures, and time series of EL.
#python
# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os 

BASE = "CREDIT_RISK_CLIENT_DATA"

def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


def load_csv(path):
    if not file_exists(path):
        st.error(f"Missing CSV: {path}")
        st.stop()
    return pd.read_csv(path, sep=";")

@st.cache_resource
def load_models(pd_path, lgd_path):
    if not file_exists(pd_path):
        raise FileNotFoundError(f"PD model not found: {pd_path}")
    if not file_exists(lgd_path):
        raise FileNotFoundError(f"LGD model not found: {lgd_path}")
    return joblib.load(pd_path), joblib.load(lgd_path)

df = load_csv(os.path.join(BASE, "features.csv"))
pd_model, lgd_model = load_models(os.path.join("pd_model.joblib"),
                                  os.path.join("lgd_model.joblib"))


st.title("Credit Risk Monitoring Dashboard")
BASE="CREDIT_RISK_CLIENT_DATA"

df = pd.read_csv(f"{BASE}/features.csv", sep=";")
pd_model = joblib.load(f"pd_model.joblib")
lgd_model = joblib.load(f"lgd_model.joblib")

features = ["LoanAmount_Cr","InterestRate","Tenure_Months","LTV","DPD","CreditScore_norm","TradeVolume_Cr"]
X = df[features].fillna(0)
df["PD"] = pd_model.predict_proba(X)[:,1]
df["LGD"] = lgd_model.predict(X).clip(0,1)
df["EAD"] = df["OutstandingBalance_Cr"]
df["EL"] = df["PD"] * df["LGD"] * df["EAD"]

st.metric("Portfolio Expected Loss", f"{df['EL'].sum():.2f} Cr")
st.metric("Weighted PD", f"{(df['PD']*df['EAD']).sum()/df['EAD'].sum():.2%}")

st.subheader("Top 10 Client Exposures")
top = df.groupby("ClientID")["EAD"].sum().sort_values(ascending=False).head(10).reset_index()
st.table(top)

st.subheader("EL Distribution")
fig, ax = plt.subplots()
ax.hist(df["EL"].clip(0,10), bins=50)
st.pyplot(fig)
