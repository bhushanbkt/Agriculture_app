import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import pickle
import cnn_model
from googletrans import Translator

# Load the trained crop recommendation model
crop_recommendation_model_path = 'RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Load the trained plant disease prediction model
model = cnn_model.CNN(39)
model.load_state_dict(torch.load("disease_model.pt"))
model.eval()

# Load disease information from CSV
disease_info_path = "disease_info.csv"
disease_info = pd.read_csv(disease_info_path, encoding='cp1252')

# Home Page
def home():
    st.title("Agriculture App")
    st.sidebar.image('farmer-148325_1280.png')
    st.write("Welcome to the Agriculture App!")
    st.write("Choose an option from the menu.")

    menu = ["Home", "Crop Recommendation", "Plant Disease Prediction"]
    choice = st.selectbox("Navigation", menu)

    if choice == "Crop Recommendation":
        crop_recommendation()
    elif choice == "Plant Disease Prediction":
        plant_disease_prediction()

# Crop Recommendation Page
def crop_recommendation():
    st.title("Crop Recommendation")
    st.subheader("Crop Recommendation Inputs")
    st.sidebar.image('lightning-9075.gif', width=300)
    N = st.slider("Nitrogen (N)", min_value=0, max_value=100, step=1, value=50)
    P = st.slider("Phosphorous (P)", min_value=0, max_value=100, step=1, value=50)
    K = st.slider("Potassium (K)", min_value=0, max_value=100, step=1, value=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=40.0, step=0.1, value=25.0)
    humidity = st.slider("Humidity", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    ph = st.slider("pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)
    rainfall = st.slider("Rainfall", min_value=0.0, max_value=500.0, step=0.1, value=100.0)

    # Display user inputs for crop recommendation
    st.sidebar.write(f"Nitrogen (N): {N}")
    st.sidebar.write(f"Phosphorous (P): {P}")
    st.sidebar.write(f"Potassium (K): {K}")
    st.sidebar.write(f"Temperature: {temperature}")
    st.sidebar.write(f"Humidity: {humidity}")
    st.sidebar.write(f"pH: {ph}")
    st.sidebar.write(f"Rainfall: {rainfall}")

    # Crop Recommendation Prediction Logic

    if st.button("Predict Crop"):
        crop_input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        predicted_crop = crop_recommendation_model.predict(crop_input_data)[0]

        # Display crop recommendation result
        st.subheader("Crop Recommendation Result")
        st.success(f"We recommend planting: {predicted_crop}")

        # Function to translate text to Hindi
        def translate_to_hindi(text):
            translator = Translator()
            translated_text = translator.translate(text, dest='hi').text
            return translated_text

        # Button to trigger translation
        if st.success("Translate to Hindi"):
            translated_recommendation = translate_to_hindi(f"We recommend planting: {predicted_crop}")
            st.success(f"Translated Recommendation (in Hindi): {translated_recommendation}")


# Plant Disease Prediction Page
def plant_disease_prediction():
    st.title("Plant Disease Prediction")
    st.sidebar.subheader("Inputs")
    st.sidebar.image('lightning-9075.gif', width=300)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Display user inputs for plant disease prediction
    if uploaded_file is not None:
        st.subheader("User Inputs for Disease Prediction")
        # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Plant Disease Prediction Logic
    if uploaded_file is not None and st.button("Predict Disease"):
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("Classifying...")

        def predict_plant_disease(image):
            # Function for plant disease prediction
            image = image.resize((224, 224))
            input_data = TF.to_tensor(image)
            input_data = input_data.view((-1, 3, 224, 224))

            with torch.no_grad():
                output = model(input_data)
                output = output.detach().numpy()
                index = np.argmax(output)

            return index

        # Predict disease index
        prediction_index = predict_plant_disease(image)
        predicted_disease = disease_info.loc[prediction_index, 'disease_name']

        if "Healthy" in predicted_disease:
            st.success("The plant is healthy.")
        else:
            # Fetch additional information from the CSV based on the predicted disease
            disease_info_row = disease_info[disease_info['disease_name'] == predicted_disease].iloc[0]

            # Assuming Translator is an object of googletrans.Translator
            translator = Translator()

            # Make sure to pass the text to be translated as an argument to the Translate method
            translated_description = translator.translate(disease_info_row['description'], dest='hi').text
            translated_steps = translator.translate(disease_info_row['Possible Steps'], dest='hi').text

            st.success(f"The plant is predicted to: {predicted_disease}")
            st.info(f"**Description (in Hindi):**\n{translated_description}")
            st.info(f"**Possible Steps (in Hindi):**\n{translated_steps}")



# Run the app
if __name__ == "__main__":
    home()