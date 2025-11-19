from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

st.header("Seksun")
st.image("img/seksun.jpg")

st.header("โปรเจคการจำแนกข้อมูลดอกไม้")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("Versicolor")
   st.image("./img/iris1.jpg")

with col2:
   st.header("Verginiga")
   st.image("./img/iris2.jpg")

with col3:
   st.header("Setosa")
   st.image("./img/iris3.jpg")

st.header("ข้อมูลดอกไม้")
st.write("ข้อมูลดอกไม้ Iris เป็นชุดข้อมูลที่มีชื่อเสียงในด้านการเรียนรู้ของเครื่อง (Machine Learning) และสถิติ ชุดข้อมูลนี้ประกอบด้วยตัวอย่างของดอกไม้ Iris ซึ่งมีสามชนิดหลัก ได้แก่ Iris setosa, Iris versicolor, และ Iris virginica")
st.write("ข้อมูลนี้มีลักษณะทางกายภาพของดอกไม้ เช่น ความยาวและความกว้างของกลีบดอกไม้ (petal) และกลีบดอก (sepal) โดยมีคุณสมบัติทั้งหมด 4 ตัวแปร ได้แก่ sepal length, sepal width, petal length, และ petal width")
st.write("ข้อมูลนี้มักถูกใช้ในการทดสอบและเปรียบเทียบอัลกอริทึมการจำแนกประเภท (classification algorithms) เนื่องจากมีความซับซ้อนที่เหมาะสมและสามารถแยกแยะชนิดของดอกไม้ได้อย่างชัดเจน")