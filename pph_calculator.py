import streamlit as st
import dill as pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="PPH Risk Calculator - Cesarean Delivery",
    page_icon="🏥",
    layout="centered"
)

# ===================== Load TabPFN Model =====================
@st.cache_resource
def load_model():
    # Please replace with your actual TabPFN model file path
    model_path = Path("cesarean_tabpfn_model.pkl")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data["model_calibrated"]
fill_dict = model_data["fill_dict"]
scaler = model_data["scaler"]  # Remove this line if TabPFN was trained without scaling
num_cols = model_data["num_cols"]
cat_cols = model_data["cat_cols"]
feature_names = model_data["feature_names"]
thresh_low = model_data["threshold_low"]
thresh_high = model_data["threshold_high"]

# ===================== Page Header =====================
st.title("Postpartum Hemorrhage Risk Prediction Calculator")
st.caption("Cesarean Delivery Cohort | TabPFN Model | External Validation AUC = 0.767")
st.markdown("---")

st.markdown("""
**Instructions for Use**
1.  Input preoperative baseline characteristics of the patient, then click *Calculate PPH Risk*.
2.  The model returns the predicted probability of postpartum hemorrhage (PPH) with three-level risk stratification.
3.  This tool is for clinical research reference only and does not replace professional medical judgment.
""")
st.markdown("---")

# ===================== Feature Display Configuration =====================
# 统一管理特征显示名称、单位、数值范围，和训练集参数完全对齐
FEATURE_LABELS = {
    "admission_age": ("Maternal Age", "years"),
    "bmi": ("Body Mass Index", "kg/m²"),
    "map": ("Mean Arterial Pressure", "mmHg"),
    "hemoglobin": ("Hemoglobin", "g/L"),
    "rbc_count": ("Red Blood Cell Count", "×10¹²/L"),
    "platelet": ("Platelet Count", "×10⁹/L"),
    "anemia": ("Anemia", ""),
    "scar_uterus": ("Scarred Uterus", ""),
    "placenta_previa": ("Placenta Previa", ""),
    "placenta_accreta": ("Placenta Accreta", ""),
    "gestational_hypertension": ("Gestational Hypertension", ""),
    "gestational_diabetes": ("Gestational Diabetes Mellitus", ""),
    "prothrombotic_state": ("Prothrombotic State", ""),
    "prolonged_labor": ("Prolonged Labor", ""),
    "malpresentation": ("Malpresentation", ""),
    "polyhydramnios": ("Polyhydramnios", ""),
    "macrosomia": ("Macrosomia", ""),
    "bad_obstetric_history": ("Adverse Obstetric History", ""),
    "intrauterine_death": ("History of Intrauterine Fetal Demise", ""),
    "thrombocytopenia": ("Thrombocytopenia", ""),
    "premature_rupture": ("Premature Rupture of Membranes", ""),
}

# 数值型特征的默认值、范围、步长，与原版本完全一致
NUMERIC_CONFIG = {
    "admission_age": {"value": 30, "min": 18, "max": 50, "step": 1},
    "bmi": {"value": 25.0, "min": 15.0, "max": 50.0, "step": 0.1},
    "map": {"value": 90, "min": 50, "max": 180, "step": 1},
    "hemoglobin": {"value": 120, "min": 50, "max": 180, "step": 1},
    "rbc_count": {"value": 4.0, "min": 2.0, "max": 6.0, "step": 0.1},
    "platelet": {"value": 200, "min": 30, "max": 500, "step": 1},
}
# ===================== Sidebar: Dynamic Input Generation =====================
st.sidebar.header("Patient Baseline Characteristics")
user_input = {}

# 第一组：数值型特征（检验与生命体征）
st.sidebar.subheader("Laboratory & Vital Signs")
for feat in num_cols:
    if feat not in feature_names:
        continue  # 模型未选用的特征，直接跳过不显示
    label_text, unit = FEATURE_LABELS.get(feat, (feat.replace("_", " ").title(), ""))
    full_label = f"{label_text} ({unit})" if unit else label_text
    config = NUMERIC_CONFIG.get(feat, {"value": 0, "min": 0, "max": 999, "step": 1})
    user_input[feat] = st.sidebar.number_input(
        full_label,
        min_value=config["min"],
        max_value=config["max"],
        value=config["value"],
        step=config["step"]
    )

# 第二组：二分类特征（产科合并症与病史）
st.sidebar.subheader("Obstetric Comorbidities & History")
for feat in cat_cols:
    if feat not in feature_names:
        continue  # 模型未选用的特征，直接跳过不显示
    label_text, _ = FEATURE_LABELS.get(feat, (feat.replace("_", " ").title(), ""))
    select_val = st.sidebar.selectbox(label_text, ["No", "Yes"])
    user_input[feat] = 1 if select_val == "Yes" else 0

# ===================== Prediction Logic =====================
def predict_pph(input_dict):
    # 1. 转为DataFrame，严格对齐模型特征顺序
    input_df = pd.DataFrame([input_dict])
    
    # 2. 兜底补齐缺失特征（极端情况备用，正常动态生成不会触发）
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = fill_dict.get(col, 0)
    
    # 3. 强制裁剪为模型训练时的特征顺序
    input_df = input_df[feature_names]
    
    # 4. 标准化
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # 5. 预测
    prob = model.predict_proba(input_df.values)[0, 1]
    return prob

# ===================== Result Display Area =====================
col1, col2 = st.columns([1, 1])
with col1:
    calculate = st.button("Calculate PPH Risk", use_container_width=True, type="primary")

if calculate:
    prob = predict_pph(user_input)
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    st.metric(label="Predicted Probability of PPH", value=f"{prob:.1%}")
    
    # 风险分层
    if prob < thresh_low:
        risk_level = "Low Risk"
        color = "#2e7d32"
    elif prob < thresh_high:
        risk_level = "Intermediate Risk"
        color = "#f57c00"
    else:
        risk_level = "High Risk"
        color = "#c62828"
    
    st.markdown(f"<h3 style='color:{color};'>Risk Stratification: {risk_level}</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Clinical Management Recommendations")
    
    if risk_level == "Low Risk":
        st.success("""
        **Routine Management Protocol**
        1.  Routine preoperative blood preparation and standard intraoperative vital sign monitoring.
        2.  Standard oxytocin infusion for uterine contraction prophylaxis during surgery.
        3.  Routine 2-hour postoperative monitoring of vaginal bleeding and uterine tone.
        """)
    elif risk_level == "Intermediate Risk":
        st.warning("""
        **Enhanced Monitoring Protocol**
        1.  Preoperative cross-matching with advance preparation of packed red blood cells.
        2.  Prophylactic administration of potent uterotonics (e.g., carboprost tromethamine) during surgery.
        3.  Extended postoperative monitoring with 15-minute interval assessment of blood loss and hemodynamics.
        4.  Preoperative preparation of emergency interventions: uterine massage and intrauterine balloon tamponade.
        """)
    else:
        st.error("""
        **High-Risk Alert Protocol**
        1.  Preoperative multidisciplinary team consultation (anesthesiology, blood bank, interventional radiology) with formal hemorrhage emergency plan.
        2.  Prophylactic internal iliac artery balloon placement for patients with suspected placental invasion abnormalities.
        3.  Intraoperative setup of autologous blood reinfusion system and prophylactic hemostatic pharmacotherapy.
        4.  Postoperative admission to intensive care unit for close monitoring, with vigilance for occult hemorrhage.
        """)

st.markdown("---")
st.caption("Disclaimer: This tool is developed based on retrospective clinical data and is intended exclusively for scientific research. It does not constitute medical advice. Clinical diagnosis and treatment must follow the judgment of licensed medical professionals.")