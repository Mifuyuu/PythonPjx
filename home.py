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
    .main-header {
        font-size: 50px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        text-shadow: 2px 2px 4px #00000020;
    }
    .sub-header {
        font-size: 25px !important;
        color: #31333F;
        text-align: center;
        margin-bottom: 30px;
    }
    .card-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .card-container:hover {
        transform: scale(1.05);
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