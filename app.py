import os
import pickle
import streamlit as st
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'water_potability_random_forest.pkl')
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    else:
        st.error("⚠️ Model file not found in 'models/' directory. Please check the path.")
        return None

# Streamlit Page Configuration
st.set_page_config(page_title="Water Potability Prediction", page_icon="💧")

# Page Title & Description
st.title('💧 Water Potability Prediction')
st.markdown("""
This app predicts whether a water sample is **potable (safe to drink)** based on various **chemical and physical parameters**.
Fill in the details below and click **Predict**.
""")

# Load the Model
model = load_model()

if model:
    # User Input Fields
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, help="Recommended: 6.5 - 8.5")
        hardness = st.number_input('Hardness (mg/L)', min_value=0.0, help="Max Safe: 300 mg/L")
        solids = st.number_input('Solids (mg/L)', min_value=0.0, help="Total dissolved solids")

    with col2:
        chloramines = st.number_input('Chloramines (ppm)', min_value=0.0, help="Max Safe: 4 ppm")
        sulfate = st.number_input('Sulfate (mg/L)', min_value=0.0, help="Max Safe: 250 mg/L")
        conductivity = st.number_input('Conductivity (μS/cm)', min_value=0.0, help="Max Safe: 500 μS/cm")

    with col3:
        organic_carbon = st.number_input('Organic Carbon (mg/L)', min_value=0.0, help="Max Safe: 2 mg/L")
        trihalomethanes = st.number_input('Trihalomethanes (μg/L)', min_value=0.0, help="Max Safe: 80 μg/L")
        turbidity = st.number_input('Turbidity (NTU)', min_value=0.0, help="Max Safe: 5 NTU")

    # Predict Button
    if st.button('Predict Potability', type='primary'):
        try:
            # Prepare input data with correct feature names
            input_data = pd.DataFrame({
                'ph': [ph],
                'Hardness': [hardness],
                'Solids': [solids],
                'Chloramines': [chloramines],
                'Sulfate': [sulfate],
                'Conductivity': [conductivity],
                'Organic_carbon': [organic_carbon],
                'Trihalomethanes': [trihalomethanes],
                'Turbidity': [turbidity]
            })

            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            # Display Result with Styling
            if prediction[0] == 1:
                st.markdown(f"""
                    <div style='background-color: #E1FFE4; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #4CAF50;'>✅ Water is Potable</h3>
                        <p style='color: #2E7D32;'>Safe to drink. Probability: {probability * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #FF6B6B;'>⚠️ Water is Not Potable</h3>
                        <p style='color: #D32F2F;'>Unsafe for drinking. Probability: {probability * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🚨 Error during prediction: {str(e)}")

# Expandable Section for Information
with st.expander("ℹ️ About this Predictor"):
    st.markdown("""
    **Water Potability Prediction Model**  
    This model is trained using **machine learning** to analyze key water quality parameters and predict if water is **safe to drink**.

    **Recommended Limits for Safe Water**:
    - **pH Level:** 6.5 - 8.5  
    - **Hardness:** ≤ 300 mg/L  
    - **Solids:** No strict limit  
    - **Chloramines:** ≤ 4 ppm  
    - **Sulfate:** ≤ 250 mg/L  
    - **Conductivity:** ≤ 500 μS/cm  
    - **Organic Carbon:** ≤ 2 mg/L  
    - **Trihalomethanes:** ≤ 80 μg/L  
    - **Turbidity:** ≤ 5 NTU  

    **How it Works**:
    - Enter the water sample values  
    - Click **Predict**  
    - The model classifies the water as **Potable (Safe) or Not Potable (Unsafe)**  

    *Disclaimer: This model is for educational purposes. For official testing, consult certified water analysis labs.*  
    """)

