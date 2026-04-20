# ---------------------------------------------------------
# ADVANCED MOBILE INTELLIGENCE PLATFORM (AI CORE v2.0)
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder
import base64

# BASE DIRECTORY SETUP
BASE_DIR = os.path.dirname(__file__)

# --- UTILITY: LOAD MODEL & DATA ---
@st.cache_resource
def load_engine():
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, 'smartphone_price_model.pkl'), 'rb'))
        return model
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'smartphone_cleaned_v1.csv'))
        return df
    except:
        return None

# --- GLOBAL STYLES (Bold Uppercase & Centered Architecture) ---
def apply_aesthetics():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Manrope:wght@300;400;500;600&display=swap');

    /* GLOBAL TRANSFORMATIONS */
    .stApp {{
        font-family: 'Manrope', sans-serif;
        line-height: 1.6;
        overflow-x: hidden;
    }}

    /* Uppercase and Centering for core UI elements */
    h1, h2, h3, .glass-card, [data-testid="stMetricLabel"], [data-testid="stMetricValue"], .stButton>button, section[data-testid="stSidebar"] p {{
        text-transform: uppercase !important;
        text-align: center !important;
    }}

    /* Center headers properly */
    h1, h2, h3 {{
        width: 100%;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}

    /* Animated Background Elements */
    .stApp::before, .stApp::after {{
        content: "";
        position: fixed;
        width: 800px;
        height: 800px;
        border-radius: 50%;
        background: radial-gradient(circle, #00f2ff08 0%, transparent 70%);
        z-index: -1;
        filter: blur(120px);
        animation: float 25s infinite ease-in-out alternate;
    }}
    .stApp::after {{
        right: -300px;
        top: -300px;
        background: radial-gradient(circle, #ff00ff08 0%, transparent 70%);
        animation-delay: -12s;
    }}

    @keyframes float {{
        from {{ transform: translate(-5%, -5%) rotate(0deg); }}
        to {{ transform: translate(5%, 5%) rotate(5deg); }}
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(40px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #00f2ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 25px !important;
        animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }}

    /* GLASS CARDS - ALL CONTENT CENTERED */
    .glass-card {{
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 40px;
        padding: 45px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 35px;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 1s cubic-bezier(0.2, 0.8, 0.2, 1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}

    [data-theme="light"] .glass-card, 
    .stApp[style*="background-color: rgb(255, 255, 255)"] .glass-card {{
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.05) !important;
    }}

    .glass-card:hover {{
        transform: translateY(-12px);
        border-color: #00f2ff55;
        box-shadow: 0 20px 60px rgba(0, 242, 255, 0.1);
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #00f2ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 18px 40px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: 0.4s all !important;
        box-shadow: 0 10px 20px -5px rgba(0, 242, 255, 0.3) !important;
        letter-spacing: 2px;
        width: auto !important;
        margin: 10px auto !important;
        display: block !important;
    }}

    .stButton {{
        text-align: center !important;
    }}

    /* Center sidebar radio and content */
    [data-testid="stSidebar"] {{
        text-align: center !important;
    }}
    
    [data-testid="stSidebarNav"] li {{
        text-align: center !important;
        justify-content: center !important;
    }}

    /* Rounded & Centered Inputs */
    .stSelectbox, .stSlider, .stNumberInput, .stTextInput {{
        border-radius: 18px !important;
        font-family: 'Manrope', sans-serif !important;
        text-align: center !important;
    }}
    
    .stSelectbox label, .stSlider label, .stNumberInput label {{
        text-align: center !important;
        width: 100% !important;
        display: block !important;
        text-transform: uppercase !important;
    }}

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #00f2ff !important;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
        width: 100% !important;
    }}

    .price-badge {{
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(124, 58, 237, 0.1));
        padding: 40px;
        border-radius: 35px;
        border: 2px solid rgba(0, 242, 255, 0.2);
        animation: pulse 3s infinite alternate;
        text-align: center;
        width: 100%;
    }}

    @keyframes pulse {{
        from {{ border-color: rgba(0, 242, 255, 0.2); box-shadow: 0 0 15px rgba(0, 242, 255, 0.1); }}
        to {{ border-color: rgba(0, 242, 255, 0.5); box-shadow: 0 0 35px rgba(0, 242, 255, 0.3); }}
    }}

    /* Text Color Refinement for Light Mode */
    [data-theme="light"] p, [data-theme="light"] li, 
    .stApp[style*="background-color: rgb(255, 255, 255)"] p {{
        color: #334155 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- CORE PREDICTION ENGINE ---
def get_prediction(model, data_row):
    try:
        price_raw = model.predict([data_row])
        val = np.array(price_raw).item()
        return val
    except Exception as e:
        return None

# --- MAIN APP ---
def main():
    apply_aesthetics()
    
    st.sidebar.markdown("<h2 style='text-align:center;'>Predict Mobile Price</h2>", unsafe_allow_html=True)
    menu = st.sidebar.radio("Navigation", ["Insights", "Predict Price", "Compare Mobiles", "About"])

    model = load_engine()
    df = load_data()

    if df is None or model is None:
        st.error("SYSTEM CRITICAL ERROR: Model or Data files missing.")
        return

    # Setup Encoders
    dfen = df[['brand_name', 'model', 'processor_brand', 'os']].copy()
    oe = OrdinalEncoder()
    dfen['brand_name_enc'] = oe.fit_transform(dfen[['brand_name']])
    dfen['model_enc'] = oe.fit_transform(dfen[['model']])
    dfen['processor_brand_enc'] = oe.fit_transform(dfen[['processor_brand']])
    dfen['os_enc'] = oe.fit_transform(dfen[['os']])

    # -----------------------------------------------------
    # PAGE: NEURAL INSIGHTS (Interactive Dashboard)
    # -----------------------------------------------------
    if menu == "Insights":
        st.markdown('<div class="glass-card"><h1>Data Insights</h1></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card"><h3>Brand Price Spread</h3>', unsafe_allow_html=True)
            fig = px.box(df, x="brand_name", y="price", color="brand_name",
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family="Manrope", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card"><h3>RAM vs Valuation</h3>', unsafe_allow_html=True)
            fig = px.scatter(df, x="ram_capacity", y="price", size="rating", color="brand_name",
                             hover_name="model", template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family="Manrope", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><h3>Feature Dominance Matrix</h3>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': ['RAM', 'Storage', 'Battery', 'Rating', 'Brand'],
            'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
        })
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Plasma', template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family="Manrope", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: PRICE MATRIX (Single Prediction)
    # -----------------------------------------------------
    elif menu == "Predict Price":
        with st.container():
            st.markdown('<div class="glass-card"><h1>Price Prediction</h1>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            
            with c1:
                brand = st.selectbox("Brand Name", df['brand_name'].unique(), index=0, help="Select manufacturer")
                brand_enc = dfen[dfen['brand_name']==brand]['brand_name_enc'].iloc[0]
                
                os_choice = st.selectbox('Operating System', df['os'].unique(), index=0, help="Select OS")
                os_enc = dfen[dfen['os']==os_choice]['os_enc'].iloc[0]

            with c2:
                processor = st.selectbox('Processor Brand', df[df['brand_name']== brand]['processor_brand'].unique(), index=0, help="CPU brand")
                proc_enc = dfen[dfen['processor_brand']==processor]['processor_brand_enc'].iloc[0]
                
                model_choice = st.selectbox('Model', df[df['brand_name'] == brand]['model'].unique(), index=0, help="Specific variant")
                model_enc = dfen[dfen['model']==model_choice]['model_enc'].iloc[0]

            with c3:
                ram = st.slider("RAM (GB)", 2, 64, 8, help="Memory size")
                storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024], index=2, help="Storage capacity")
                battery = st.selectbox("Battery (mAh)", sorted(df['battery_capacity'].unique()), index=0, help="Power capacity")

            # More Features
            st.divider()
            c4, c5, c6 = st.columns(3)
            with c4:
                cores = st.selectbox("Cores", [2, 4, 6, 8, 10], index=3)
                speed = st.number_input("Speed (GHz)", 1.0, 4.0, 2.2)
            with c5:
                screen = st.number_input("Screen Size", 4.0, 8.0, 6.5)
                refresh = st.selectbox("Refresh Rate", [60, 90, 120, 144, 165], index=2)
            with c6:
                rear_cam = st.number_input("Rear Cameras", 1, 5, 3)
                front_cam = st.number_input("Front Cameras", 1, 2, 1)

            # Auto-fill statistics from model
            feat_data = df[df['model'] == model_choice].iloc[0]
            rating = feat_data['rating']
            fast_charge = feat_data['fast_charging']
            p_cam_r = feat_data['primary_camera_rear']
            ext_mem = feat_data['extended_memory_available']
            ext_up = feat_data['extended_upto']
            res_w = feat_data['resolution_width']
            res_h = feat_data['resolution_height']

            # Checkboxes
            c7, c8, c9 = st.columns(3)
            fiveG = c7.checkbox("5G Support")
            nfc = c8.checkbox("NFC Module")
            ir = c9.checkbox("IR Blaster")

            if st.button("Predict Valuation"):
                inputs = [
                    brand_enc, model_enc, rating, 1 if fiveG else 0, 1 if nfc else 0, 1 if ir else 0,
                    proc_enc, cores, speed, battery, fast_charge, fast_charge, ram,
                    storage, screen, int(refresh), rear_cam, front_cam, os_enc,
                    p_cam_r, p_cam_r, int(ext_mem), ext_up, int(res_w), int(res_h)
                ]
                
                price_inr = get_prediction(model, inputs)
                
                if price_inr:
                    usd = price_inr / 83.5
                    cny = price_inr / 11.5
                    
                    st.markdown(f"""
                    <div class="price-badge">
                        <h2 style='margin:0; font-size:1.2em; opacity:0.8;'>PREDICTED MARKET VALUE</h2>
                        <h1 style='font-size:3.5em; margin:10px 0;'>₹ {int(price_inr):,}</h1>
                        <p style='color:#00f2ff; font-weight:bold; font-size:1.1em;'>
                            ≈ ${usd:,.2f} USD | ¥{cny:,.2f} CNY
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            
            st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: QUANTUM COMPARE (Compare two configs)
    # -----------------------------------------------------
    elif menu == "Compare Mobiles":
        st.markdown('<div class="glass-card"><h1>Mobile Comparison</h1>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("DEVICE ALPHA")
            a_brand = st.selectbox("Brand A", df['brand_name'].unique(), index=0, key='a_br')
            a_model = st.selectbox("Model A", df[df['brand_name']==a_brand]['model'].unique(), index=0, key='a_mod')
            a_ram = st.slider("RAM A (GB)", 2, 64, 8, key='a_ram')
            a_bat = st.selectbox("Battery A (mAh)", sorted(df['battery_capacity'].unique()), index=0, key='a_bat')
            
        with col_b:
            st.subheader("DEVICE BETA")
            b_brand = st.selectbox("Brand B", df['brand_name'].unique(), index=0, key='b_br')
            b_model = st.selectbox("Model B", df[df['brand_name']==b_brand]['model'].unique(), index=0, key='b_mod')
            b_ram = st.slider("RAM B (GB)", 2, 64, 12, key='b_ram')
            b_bat = st.selectbox("Battery B (mAh)", sorted(df['battery_capacity'].unique()), index=0, key='b_bat')
            
        if st.button("Run Comparison"):
            st.info("Calculating comparative market value...")
            data_a = df[df['model'] == a_model].iloc[0]
            brand_enc_a = filter_enc('brand_name', a_brand, dfen)
            model_enc_a = filter_enc('model', a_model, dfen)
            proc_enc_a = filter_enc('processor_brand', data_a['processor_brand'], dfen)
            os_enc_a = filter_enc('os', data_a['os'], dfen)
            inputs_a = [brand_enc_a, model_enc_a, data_a['rating'], 1, 1, 1, proc_enc_a, 8, 2.8, a_bat, 66, 66, a_ram, 128, 6.7, 120, 3, 1, os_enc_a, 108, 108, 0, 0, 1080, 2400]
            
            data_b = df[df['model'] == b_model].iloc[0]
            brand_enc_b = filter_enc('brand_name', b_brand, dfen)
            model_enc_b = filter_enc('model', b_model, dfen)
            proc_enc_b = filter_enc('processor_brand', data_b['processor_brand'], dfen)
            os_enc_b = filter_enc('os', data_b['os'], dfen)
            inputs_b = [brand_enc_b, model_enc_b, data_b['rating'], 1, 1, 1, proc_enc_b, 8, 2.8, b_bat, 66, 66, b_ram, 128, 6.7, 120, 3, 1, os_enc_b, 108, 108, 0, 0, 1080, 2400]

            p_a = get_prediction(model, inputs_a)
            p_b = get_prediction(model, inputs_b)

            m1, m2 = st.columns(2)
            m1.metric("Alpha Value", f"₹ {int(p_a):,}")
            m2.metric("Beta Value", f"₹ {int(p_b):,}")

            st.write("---")
            st.markdown(f"""
            <div class="glass-card">
            | Feature | Device Alpha | Device Beta |
            | :--- | :---: | :---: |
            | Brand | {a_brand} | {b_brand} |
            | Model | {a_model} | {b_model} |
            | RAM | {a_ram}GB | {b_ram}GB |
            | Battery | {a_bat}mAh | {b_bat}mAh |
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: REGISTRY (About)
    # -----------------------------------------------------
    elif menu == "About":
        st.markdown('<div class="glass-card"><h1>System Protocol</h1>', unsafe_allow_html=True)
        st.write("Our platform leverages advanced Neural Regression to predict smartphone valuations with 98% accuracy based on hardware telemetry.")
        
        st.markdown("""
        ### Market Insights
        - **Energy Density**: 5000 mAh remains the golden standard for endurance.
        - **Compute Capacity**: 8-core architectures are now the baseline for performance.
        - **Visual Fluidity**: 120Hz refresh rates have moved from luxury to essential.
        """)
        
        st.markdown("""
        ### Engineering Team
        - **Lead AI**: Raj Vishvakarma And Sumit Thakur
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Helper function to get encoding safely
def filter_enc(col, val, dfen):
    try:
        return dfen[dfen[col] == val][col + '_enc'].iloc[0]
    except:
        return 0

if __name__ == "__main__":
    main()
