import streamlit as st

st.set_page_config(page_title="AI Internship Demo", page_icon="🤖", layout="wide")

st.title("🤖 AI Internship Project Demos")
st.caption("Mohit Agrawal · Roll No. 51 · B.Tech (Electronics & Computer Science), RCOEM Nagpur · "
           "AI Intern, Codec Technologies")

st.markdown(
    "Two end-to-end projects built during the 8-week internship, covering both paradigms of "
    "applied AI: **classical ML on tabular data** and **deep learning on images**."
)

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.subheader("📉 Project 1 · Customer Churn Predictor")
        st.write("Scikit-Learn pipeline · SMOTE (inside CV folds) · Random Forest · "
                 "recall-oriented decision threshold.")
        st.page_link("pages/1_Churn_Predictor.py", label="Open the churn demo", icon="➡️")
with c2:
    with st.container(border=True):
        st.subheader("🖼️ Project 2 · CNN Image Classifier")
        st.write("TensorFlow/Keras CNN · Conv2D + MaxPooling · Dropout · Adam · "
                 "Categorical Cross-Entropy · EarlyStopping.")
        st.page_link("pages/2_Image_Classifier.py", label="Open the image classifier", icon="➡️")

st.divider()
st.caption("Use the sidebar to switch between projects.")
