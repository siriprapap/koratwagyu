import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

st.set_page_config(page_title="ทำนายน้ำหนักวัว", layout="centered")

# ส่วนหัวเว็บ
st.title("🐄 ทำนายน้ำหนักวัวจากภาพ")
st.write("อัปโหลดภาพวัว แล้วระบบจะคาดการณ์น้ำหนักให้คุณทันที!")

# โหลดโมเดล (ตรวจสอบว่ามีไฟล์อยู่)
MODEL_PATH = 'KoratCattle.h5'

if not os.path.exists(MODEL_PATH):
    st.error(f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' กรุณาอัปโหลดไว้ในโฟลเดอร์เดียวกับ app.py")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

    # อัปโหลดภาพ
    uploaded_file = st.file_uploader("เลือกรูปภาพวัว (jpg, png)", type=["jpg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='ภาพที่อัปโหลด', use_column_width=True)

            # ปรับภาพ
            image = image.resize((224, 224)).convert('L')  # Grayscale
            img = np.array(image) / 255.0
            img = img.reshape(1, 224, 224, 1)

            # ทำนาย
            prediction = model.predict(img)
            predicted_weight = prediction[0][0]

            # แสดงผลลัพธ์
            st.success(f"🎯 น้ำหนักที่คาดการณ์: **{predicted_weight:.2f} กิโลกรัม**")

        except UnidentifiedImageError:
            st.error("ไม่สามารถเปิดไฟล์ภาพได้ กรุณาเลือกรูปภาพที่ถูกต้อง (jpg, png)")
