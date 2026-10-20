from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "mobile_kmeans.joblib"

# ลำดับฟีเจอร์ต้องตรงกับตอนเทรนโมเดล
FEATURES = ["ram", "battery_power", "px_width", "px_height", "mobile_wt"]
FEATURE_TH = {
    "ram": "RAM (MB)",
    "battery_power": "แบตเตอรี่ (mAh)",
    "px_width": "จอกว้าง (px)",
    "px_height": "จอสูง (px)",
    "mobile_wt": "น้ำหนัก (g)",
}

# ชื่อกลุ่ม ตั้งจากค่าเฉลี่ยสเปกของแต่ละกลุ่ม (ตารางที่ 2 ในบทความ; Code/V3/results/table2_cluster_profile.csv)
# โมเดล mobile_kmeans.joblib บันทึกจาก Code/V3/analysis_final.ipynb (Pipeline: StandardScaler -> K-Means, k=4, random_state=42)
CLUSTER_INFO = {
    0: ("กลุ่มจอคมชัด", "ความละเอียดหน้าจอสูงที่สุด RAM ค่อนข้างสูง"),
    1: ("กลุ่มเครื่องเบา สเปกพื้นฐาน", "น้ำหนักเบาที่สุด RAM ปานกลาง จอความละเอียดต่ำ"),
    2: ("กลุ่มเครื่องหนัก สเปกต่ำ", "น้ำหนักมากที่สุด RAM ต่ำที่สุด"),
    3: ("กลุ่ม RAM สูง (สเปกแรง)", "RAM สูงที่สุด ความละเอียดหน้าจอต่ำที่สุด"),
}

# ชื่อสั้นสำหรับแกนของแผนภูมิ
SHORT_NAME = {0: "0 จอคมชัด", 1: "1 เครื่องเบา", 2: "2 เครื่องหนัก", 3: "3 RAM สูง"}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="จัดกลุ่มมือถือ (K-Means)", page_icon="📱")
st.title("📱 จัดกลุ่มมือถือตามสเปก")


if not MODEL_PATH.exists():
    st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH.name} (ต้องวางไว้โฟลเดอร์เดียวกับ app.py)")
    st.stop()

model = load_model()
scaler = model.named_steps["scaler"]
kmeans = model.named_steps["kmeans"]

# จุดศูนย์กลางของแต่ละกลุ่ม แปลงกลับเป็นหน่วยจริง
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES
).round(0)
centers.index = [f"{i}: {CLUSTER_INFO[i][0]}" for i in range(len(centers))]

st.subheader("กรอกสเปกมือถือ")
col1, col2 = st.columns(2)

with col1:
    ram = st.number_input("RAM (MB)", min_value=256, max_value=4000, value=2000, step=64)
    battery_power = st.number_input(
        "ความจุแบตเตอรี่ (mAh)", min_value=500, max_value=2000, value=1200, step=50
    )
    mobile_wt = st.number_input(
        "น้ำหนักเครื่อง (กรัม)", min_value=80, max_value=200, value=140, step=1
    )

with col2:
    px_width = st.number_input(
        "ความละเอียดหน้าจอ - กว้าง (px)", min_value=500, max_value=2000, value=1000, step=10
    )
    px_height = st.number_input(
        "ความละเอียดหน้าจอ - สูง (px)", min_value=0, max_value=2000, value=700, step=10
    )

if st.button("จัดกลุ่ม", type="primary"):
    X = pd.DataFrame(
        [[ram, battery_power, px_width, px_height, mobile_wt]], columns=FEATURES
    )
    cluster = int(model.predict(X)[0])
    name, desc = CLUSTER_INFO.get(cluster, (f"กลุ่ม {cluster}", ""))

    st.success(f"มือถือเครื่องนี้อยู่ **กลุ่ม {cluster} – {name}**")
    st.write(desc)

    # ระยะห่างจากศูนย์กลางแต่ละกลุ่ม (ยิ่งน้อยยิ่งใกล้)
    dist = model.transform(X)[0]
    # ใช้ชื่อกลุ่มแบบสั้นบนแกนของกราฟ เพื่อไม่ให้ชื่อถูกตัด
    dist_df = pd.DataFrame(
        {"ระยะห่างจากศูนย์กลางกลุ่ม": dist}, index=[SHORT_NAME[i] for i in range(len(dist))]
    )
    st.subheader("ระยะห่างจากศูนย์กลางของแต่ละกลุ่ม (ยิ่งน้อยยิ่งใกล้)")
    st.bar_chart(dist_df, horizontal=True)

    with st.expander("ข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(X.rename(columns=FEATURE_TH))

st.subheader("ลักษณะของแต่ละกลุ่ม (ค่าเฉลี่ย / จุดศูนย์กลาง)")
st.dataframe(centers.rename(columns=FEATURE_TH))
