

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os 

os.chdir(r'C:\GYANENDRA\INFORMATION_TECHNILOGY_PROJECTS\CREDIT_RISK_ENGINE')

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
