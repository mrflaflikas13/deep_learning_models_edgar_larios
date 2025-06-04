import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["About", "Text Classification", "Image Classification", "Regression"])

# ABOUT PAGE
if page == "About":
    st.title("About This Project")
    st.write("""
    This Streamlit app demonstrates 3 deep learning models:
    - Text classification (e-commerce product categories)
    - Image classification (butterflies)
    - Regression (Walmart sales)

    All models were trained using custom datasets and saved using Keras and pickle.
    """)

# TEXT CLASSIFICATION PAGE
elif page == "Text Classification":
    st.title("Text Classification Model")

    # Load model and helpers
    model = load_model("model_text.keras")
    with open("tokenizer_text.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder_text.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    input_text = st.text_input("Enter text:")
    if st.button("Predict"):
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        prediction = model.predict(padded)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([class_index])[0]
        st.write(f"Predicted Category: {predicted_label}")

# IMAGE CLASSIFICATION PAGE
elif page == "Image Classification":
    st.title("Image Classification Model")

    # Load model and label encoder
    model = load_model("model_image.h5")
    with open("label_encoder_image.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    uploaded_file = st.file_uploader("Upload a butterfly image", type=["jpg", "png"])
    if uploaded_file:
        image = load_img(uploaded_file, target_size=(128, 128))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([class_index])[0]
        st.write(f"Predicted Butterfly Species: {predicted_label}")

# REGRESSION PAGE
elif page == "Regression":
    st.title("Weekly Sales Regression Model")

    # Load model and scaler
    model = load_model("model_regression.h5")
    with open("scaler_regression.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.write("Enter the 10 numeric input features below:")
    features = []
    feature_names = ["Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI",
                     "Unemployment", "Year", "Month", "Day", "WeekOfYear"]

    for name in feature_names:
        val = st.number_input(f"{name}", step=1.0, format="%.2f")
        features.append(val)

    if st.button("Predict Sales"):
        input_array = np.array([features])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0][0]
        st.write(f"Predicted Weekly Sales: ${prediction:,.2f}")

