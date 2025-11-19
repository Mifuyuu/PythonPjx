import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บ (ต้องอยู่บรรทัดแรกๆ ของ Streamlit)
st.set_page_config(
    page_title="Iris Classification Project",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS เพื่อความสวยงาม
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }

    .main-header {
        font-size: 48px !important;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    
    .sub-header {
        font-size: 24px !important;
        font-weight: 300;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .card-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    
    .card-container:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        border-color: #FF4B4B;
    }
    
    img {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ส่วน Sidebar
with st.sidebar:
    st.image("img/seksun.jpg", caption="Developer Profile", use_column_width=True)
    st.title("Seksun")
    st.info("ผู้พัฒนาโปรเจคจำแนกสายพันธุ์ดอกไม้ด้วย Machine Learning")

# ส่วนเนื้อหาหลัก
st.markdown('<p class="main-header">🌸 โปรเจคการจำแนกข้อมูลดอกไม้ 🌸</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบอัจฉริยะสำหรับจำแนกสายพันธุ์ดอก Iris</p>', unsafe_allow_html=True)

st.divider()

# แสดงข้อมูลดอกไม้ 3 สายพันธุ์
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)
    st.markdown("### 💜 Versicolor")
    st.caption("ดอกไอริสเวอร์สิคัลเลอร์")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)
    st.markdown("### 💙 Virginica")
    st.caption("ดอกไอริสเวอร์จินิกา")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)
    st.markdown("### 💗 Setosa")
    st.caption("ดอกไอริสเซโตซา")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.success("หน้าเว็บพร้อมใช้งานแล้ว!")
st.balloons()