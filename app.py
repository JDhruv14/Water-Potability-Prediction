import os
import pickle
import streamlit as st
import pandas as pd

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'water_potability_random_forest.pkl')
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    else:
        st.error("Model file not found in 'models/' directory. Please check the path.")
        return None

# Page configuration
st.set_page_config(page_title="Water Potability Prediction", page_icon="💧")

st.title('💧 Water Potability Prediction')
st.markdown("""
This app predicts whether the given water sample is potable based on various water quality parameters.
""")

model = load_model()

if model:
    # Create layout with 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input('pH Level', help="Typical range: 6.5 - 8.5")
        hardness = st.number_input('Hardness (mg/L)')
        solids = st.number_input('Solids (mg/L)')
    
    with col2:
        chloramines = st.number_input('Chloramines (ppm)')
        sulfate = st.number_input('Sulfate (mg/L)')
        conductivity = st.number_input('Conductivity (μS/cm)')
    
    with col3:
        organic_carbon = st.number_input('Organic Carbon (mg/L)')
        trihalomethanes = st.number_input('Trihalomethanes (μg/L)')
        turbidity = st.number_input('Turbidity (NTU)')

    # Predict button
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
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]  # Probability of being potable

            # Show result with the same old UI
            if prediction == 1:
                st.markdown(f"""
                    <div style='background-color: #E1FFE4; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #6BFF6B; margin: 0;'>✅ Water is Potable</h3>
                        <p style='color: #4CAF50;'>The water is safe to drink. Probability: {probability * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #FF6B6B; margin: 0;'>⚠️ Water is Not Potable</h3>
                        <p style='color: #FF3333;'>The water is unsafe for drinking. Probability: {probability * 100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    # Expandable section for more info
    with st.expander("About this predictor"):
        st.markdown("""
        This water potability prediction model uses a machine learning algorithm trained on water quality data.
        The model evaluates water safety based on key chemical and physical properties.

        **Features used and recommended limits:**
        - **pH Level:** 6.5 - 8.5 (Safe range)
        - **Hardness:** ≤ 300 mg/L
        - **Solids:** No strict limit, total dissolved solids
        - **Chloramines:** ≤ 4 ppm
        - **Sulfate:** ≤ 250 mg/L
        - **Conductivity:** ≤ 500 μS/cm
        - **Organic Carbon:** ≤ 2 mg/L
        - **Trihalomethanes:** ≤ 80 μg/L
        - **Turbidity:** ≤ 5 NTU
        """)
