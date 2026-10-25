import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# 页面配置
st.set_page_config(page_title="Cesarean PPH Risk Calculator", page_icon="🏥", layout="wide")

st.title("🏥 Cesarean Delivery Postpartum Hemorrhage (PPH) Risk Calculator")
st.markdown("""
**Developed based on a LightGBM model (n=4,270).** 
*This tool is intended for **research and educational purposes only**. It must not be used as the sole basis for clinical decision-making.*
""")
st.markdown("---")

# 侧边栏：输入特征
st.sidebar.header("Patient Clinical Features")

@st.cache_resource
def load_model():
    # 请确保模型路径正确，如果是云端部署，建议将模型放在同目录下
    MODEL_PATH = "cesarean_lightgbm_model.pkl"
    try:
        model_obj = joblib.load(MODEL_PATH)
        return model_obj if not isinstance(model_obj, dict) else model_obj["final_model"]
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

clf = load_model()

if clf is not None:
    st.sidebar.subheader("1. Continuous Variables")
    bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=30.9, step=0.1)
    map_val = st.sidebar.number_input("Mean Arterial Pressure (mmHg)", min_value=40.0, max_value=150.0, value=87.3, step=1.0)
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=4.0, max_value=20.0, value=11.8, step=0.1)
    rbc = st.sidebar.number_input("RBC Count (×10¹²/L)", min_value=1.5, max_value=7.0, value=4.0, step=0.1)
    wbc = st.sidebar.number_input("WBC Count (×10⁹/L)", min_value=1.0, max_value=40.0, value=10.8, step=0.1)
    plt_val = st.sidebar.number_input("Platelet Count (×10⁹/L)", min_value=10.0, max_value=600.0, value=219.0, step=1.0)

    st.sidebar.subheader("2. Maternal & Obstetric Factors")
    scar = st.sidebar.checkbox("Scarred Uterus")
    accreta = st.sidebar.checkbox("Placenta Accreta")
    previa = st.sidebar.checkbox("Placenta Previa")
    poly = st.sidebar.checkbox("Polyhydramnios")
    gdm = st.sidebar.checkbox("Gestational Diabetes (GDM)")
    anemia = st.sidebar.checkbox("Anemia")
    iud = st.sidebar.checkbox("Intrauterine Death")
    prolabor = st.sidebar.checkbox("Prolonged Labor")
    malpres = st.sidebar.checkbox("Malpresentation")
    prothrom = st.sidebar.checkbox("Prothrombotic State / APS")
    prom = st.sidebar.checkbox("Premature Rupture of Membranes")
    oxytocin = st.sidebar.checkbox("Oxytocin use (Pre-48h)")
    pe = st.sidebar.checkbox("Preeclampsia")

    # 主界面
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Summary")
        st.write(f"**BMI:** {bmi} kg/m²")
        st.write(f"**MAP:** {map_val} mmHg")
        st.write(f"**Hb:** {hemoglobin} g/dL")
        st.write(f"**PLT:** {plt_val} ×10⁹/L")
        
    with col2:
        st.subheader("Prediction Result")
        if st.button("Calculate PPH Risk", type="primary", use_container_width=True):
            input_df = pd.DataFrame([{
                'bmi': bmi, 'map': map_val, 'hemoglobin': hemoglobin, 'rbc_count': rbc,
                'wbc_count': wbc, 'platelet': plt_val,
                'scar_uterus': int(scar), 'placenta_accreta': int(accreta),
                'placenta_previa': int(previa), 'polyhydramnios': int(poly),
                'gestational_diabetes': int(gdm), 'anemia': int(anemia),
                'intrauterine_death': int(iud), 'prolonged_labor': int(prolabor),
                'malpresentation': int(malpres), 'prothrombotic_state': int(prothrom),
                'premature_rupture': int(prom), 'oxytocin_pre48h': int(oxytocin),
                'preeclampsia': int(pe)
            }])
            
            # 预测并输出
            prob = clf.predict_proba(input_df)[0, 1]
            prob_pct = prob * 100
            
            # 使用 INSPIRE 外部验证确定的最优阈值 0.102 进行分层
            if prob < 0.05:
                st.success(f"### Predicted Risk: {prob_pct:.1f}% (Low Risk)")
                st.info("Risk is below 5%. Routine monitoring is recommended.")
            elif prob < 0.102:
                st.warning(f"### Predicted Risk: {prob_pct:.1f}% (Moderate Risk)")
                st.info("Risk is moderate. Consider enhanced postpartum observation.")
            else:
                st.error(f"### Predicted Risk: {prob_pct:.1f}% (High Risk)")
                st.info("Risk is high. Immediate preparation for hemorrhage management is advised.")
                
            st.markdown("""
            > **Note:** In our external validation (INSPIRE cohort, n=1,667), the model achieved an AUC of 0.640. 
            > The optimal threshold was 0.102. The risk stratification is based on this threshold.
            """)
