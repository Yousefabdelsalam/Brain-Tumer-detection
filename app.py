
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# إعداد الصفحة
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تصميم CSS بسيط لتجميل الصفحة
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #5A189A;
        font-size: 42px;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        color: #7B2CBF;
        font-size: 22px;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# تحميل الموديل
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# الشريط الجانبي (Sidebar)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Detection", "ℹ️ About"])

# ------------------------------------------------------------
# 🏠 الصفحة الرئيسية
# ------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🧠 Brain Tumor Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Detect different types of brain tumors using YOLOv8 deep learning model</p>", unsafe_allow_html=True)

    st.image("Brain-tumor-severity-and-survival-may-be-dependent-on-sex-scaled.jpg", use_column_width=True)
    st.write("""
    ---
    ### ⚙️ Features:
    - Upload **multiple MRI images**
    - Detect **Glioma, Meningioma, Pituitary**, and **No Tumor**
    - View detection **confidence scores**
    - **Modern interface** with clear results visualization
    ---
    """)

# ------------------------------------------------------------
# 🔍 صفحة الكشف
# ------------------------------------------------------------
elif page == "🔍 Detection":
    st.markdown("<h1 class='main-title'>🔍 Brain Tumor Detection</h1>", unsafe_allow_html=True)
    st.success("✅ Model loaded successfully!")

    uploaded_files = st.file_uploader(
        "Upload one or more MRI images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📸 You uploaded {len(uploaded_files)} image(s).")
        total_detected = 0

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption=f"🩻 Original: {uploaded_file.name}", width=350)

            # الكشف
            results = model.predict(image, conf=0.5)
            result_img = results[0].plot()
            boxes = results[0].boxes

            with col2:
                st.image(result_img, caption="🎯 Detection Result", width=350)

                if len(boxes) == 0:
                    st.warning("No tumor detected.")
                else:
                    total_detected += len(boxes)
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.write("**Detection Summary:**")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"- 🧩 **Class:** {model.names[cls]} | 🎯 **Confidence:** {conf:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

        # إحصائيات سريعة
        st.markdown(f"### 📊 Total Detected Tumors: **{total_detected}**")

# ------------------------------------------------------------
# ℹ️ صفحة حول المشروع
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-title'>ℹ️ About the Project</h1>", unsafe_allow_html=True)

    st.write("""
    This project is built using **YOLOv8** (You Only Look Once), a real-time object detection model,
    to automatically detect and classify brain tumors in MRI images.

    ### 🧩 Classes Detected:
    - **Glioma**
    - **Meningioma**
    - **Pituitary**
    - **No Tumor**

    ### 🧠 Workflow:
    1. Data collected from **Roboflow**
    2. Preprocessing and augmentation performed
    3. Model trained with YOLOv8
    4. Evaluation on test data (`mAP50 = 0.93`)
    5. Deployment using **Streamlit**
    
    ---
    **Developer:** AI Working Yousef 👨‍💻  
    **Frameworks:** Ultralytics, Streamlit, PyTorch  
    **Version:** 1.0.0
    """)

