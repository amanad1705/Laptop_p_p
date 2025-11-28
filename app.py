# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Laptop Price Predictor 💻",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== LOAD MODEL AND FEATURES ==========
@st.cache_resource
def load_model_data():
    """Load the trained model and feature columns"""
    try:
        model = pickle.load(open("model.pkl", "rb"))
        features = pickle.load(open("features.pkl", "rb"))
        return model, features
    except FileNotFoundError as e:
        st.error(f"❌ Error: {e}. Please ensure model.pkl and features.pkl are in the same directory.")
        st.stop()

model, features = load_model_data()

# ========== DATA LISTS FROM YOUR DATASET ==========
CPU_COMPANIES = ['Intel', 'AMD', 'Samsung']

CPU_MODELS = [
    'Core i5', 'Core i5 7200U', 'Core i7', 'A9-Series 9420', 'Core i7 8550U',
    'Core i5 8250U', 'Core i3 6006U', 'Core M m3', 'Core i7 7500U',
    'Core i3 7100U', 'Atom x5-Z8350', 'Core i5 7300HQ', 'E-Series E2-9000e',
    'Core i7 8650U', 'Atom x5-Z8300', 'E-Series E2-6110', 'A6-Series 9220',
    'Celeron Dual Core N3350', 'Core i3 7130U', 'Core i7 7700HQ', 'Ryzen 1700',
    'Pentium Quad Core N4200', 'Atom x5-Z8550', 'Celeron Dual Core N3060',
    'FX 9830P', 'Core i7 7560U', 'E-Series 6110', 'Core i5 6200U', 'Core M 6Y75',
    'Core i5 7500U', 'Core i7 6920HQ', 'Core i5 7Y54', 'Core i7 7820HK',
    'Xeon E3-1505M V6', 'Core i7 6500U', 'E-Series 9000e',
    'A10-Series A10-9620P', 'A6-Series A6-9220', 'Core i7 6600U',
    'Celeron Dual Core 3205U', 'Core i7 7820HQ', 'A10-Series 9600P',
    'Core i7 7600U', 'A8-Series 7410', 'Celeron Dual Core 3855U',
    'Pentium Quad Core N3710', 'A12-Series 9720P', 'Core i5 7300U',
    'Celeron Quad Core N3450', 'Core i5 6440HQ', 'Core i7 6820HQ', 'Ryzen 1600',
    'Core i7 7Y75', 'Core i5 7440HQ', 'Core i7 7660U', 'Core M m3-7Y30',
    'Core i5 7Y57', 'Core i7 6700HQ', 'Core i3 6100U', 'A10-Series 9620P',
    'E-Series 7110', 'A9-Series A9-9420', 'Core i7 6820HK', 'Core M 7Y30',
    'Xeon E3-1535M v6', 'Celeron Quad Core N3160', 'Core i5 6300U',
    'E-Series E2-9000', 'Celeron Dual Core N3050', 'Core M M3-6Y30',
    'Core i5 6300HQ', 'A6-Series 7310', 'Atom Z8350', 'Xeon E3-1535M v5',
    'Core i5 6260U', 'Pentium Dual Core N4200', 'Celeron Quad Core N3710',
    'Core M', 'A12-Series 9700P', 'Pentium Dual Core 4405U', 'A4-Series 7210',
    'Core i7 6560U', 'Core M m7-6Y75', 'FX 8800P', 'Core M M7-6Y75',
    'Atom X5-Z8350', 'Pentium Dual Core 4405Y', 'Pentium Quad Core N3700',
    'Core M 6Y54', 'Cortex A72&A53', 'E-Series 9000', 'Core M 6Y30',
    'A9-Series 9410'
]

GPU_COMPANIES = ['Intel', 'AMD', 'Nvidia', 'ARM']

GPU_MODELS = [
    'Iris Plus Graphics 640', 'HD Graphics 6000', 'HD Graphics 620',
    'Radeon Pro 455', 'Iris Plus Graphics 650', 'Radeon R5', 'Iris Pro Graphics',
    'GeForce MX150', 'UHD Graphics 620', 'HD Graphics 520', 'Radeon Pro 555',
    'Radeon R5 M430', 'HD Graphics 615', 'Radeon Pro 560', 'GeForce 940MX',
    'HD Graphics 400', 'GeForce GTX 1050', 'Radeon R2', 'Radeon 530',
    'GeForce 930MX', 'HD Graphics', 'HD Graphics 500', 'GeForce 930MX ',
    'GeForce GTX 1060', 'GeForce 150MX', 'Iris Graphics 540', 'Radeon RX 580',
    'GeForce 920MX', 'Radeon R4 Graphics', 'Radeon 520', 'GeForce GTX 1070',
    'GeForce GTX 1050 Ti', 'GeForce MX130', 'R4 Graphics', 'GeForce GTX 940MX',
    'Radeon RX 560', 'GeForce 920M', 'Radeon R7 M445', 'Radeon RX 550',
    'GeForce GTX 1050M', 'HD Graphics 515', 'Radeon R5 M420', 'HD Graphics 505',
    'GTX 980 SLI', 'R17M-M1-70', 'GeForce GTX 1080', 'Quadro M1200',
    'GeForce 920MX ', 'GeForce GTX 950M', 'FirePro W4190M ', 'GeForce GTX 980M',
    'Iris Graphics 550', 'GeForce 930M', 'HD Graphics 630', 'Radeon R5 430',
    'GeForce GTX 940M', 'HD Graphics 510', 'HD Graphics 405', 'Radeon RX 540',
    'GeForce GT 940MX', 'FirePro W5130M', 'Quadro M2200M', 'Radeon R4',
    'Quadro M620', 'Radeon R7 M460', 'HD Graphics 530', 'GeForce GTX 965M',
    'GeForce GTX1080', 'GeForce GTX1050 Ti', 'GeForce GTX 960M',
    'Radeon R2 Graphics', 'Quadro M620M', 'GeForce GTX 970M',
    'GeForce GTX 960<U+039C>', 'Graphics 620', 'GeForce GTX 960',
    'Radeon R5 520', 'Radeon R7 M440', 'Radeon R7', 'Quadro M520M',
    'Quadro M2200', 'Quadro M2000M', 'HD Graphics 540', 'Quadro M1000M',
    'Radeon 540', 'GeForce GTX 1070M', 'GeForce GTX1060', 'HD Graphics 5300',
    'Radeon R5 M420X', 'Radeon R7 Graphics', 'GeForce 920', 'GeForce 940M',
    'GeForce GTX 930MX', 'Radeon R7 M465', 'Radeon R3', 'GeForce GTX 1050Ti',
    'Radeon R7 M365X', 'Radeon R9 M385', 'HD Graphics 620 ', 'Quadro 3000M',
    'GeForce GTX 980 ', 'Radeon R5 M330', 'FirePro W4190M', 'FirePro W6150M',
    'Radeon R5 M315', 'Quadro M500M', 'Radeon R7 M360', 'Quadro M3000M',
    'GeForce 960M', 'Mali T860 MP4'
]

MEMORY_OPTIONS = [
    "128GB SSD", "256GB SSD", "512GB SSD", "1TB SSD", "2TB SSD",
    "500GB HDD", "1TB HDD", "2TB HDD",
    "128GB Flash Storage", "256GB Flash Storage",
    "128GB SSD + 1TB HDD", "256GB SSD + 1TB HDD", "512GB SSD + 1TB HDD"
]

# ========== SIDEBAR: INPUT SPECIFICATIONS ==========
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/laptop.png", width=150)
    st.title("🔧 Configure Laptop")
    st.write("Adjust specifications below")
    
    st.divider()
    
    # Display Section
    st.write("### 📏 Display")
    inches = st.slider(
        "Screen Size (inches)", 
        min_value=10.0, 
        max_value=20.0, 
        value=15.6, 
        step=0.1,
        help="Typical range: 13-17 inches"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        screen_w = st.number_input(
            "Width (px)", 
            min_value=800, 
            max_value=4000, 
            value=1920, 
            step=100
        )
    with col2:
        screen_h = st.number_input(
            "Height (px)", 
            min_value=600, 
            max_value=4000, 
            value=1080, 
            step=100
        )
    
    st.divider()
    
    # Performance Section
    st.write("### ⚡ Performance")
    
    ram = st.select_slider(
        "🧠 RAM (GB)", 
        options=[2, 4, 8, 12, 16, 24, 32, 64],
        value=8,
        help="More RAM = better multitasking"
    )
    
    # CPU Selection with search
    cpu = st.selectbox(
        "🔲 Processor (CPU)", 
        options=sorted(CPU_MODELS),
        index=sorted(CPU_MODELS).index('Core i5 7200U'),
        help="Choose your processor model"
    )
    
    # GPU Selection with search
    gpu = st.selectbox(
        "🎮 Graphics (GPU)", 
        options=sorted(GPU_MODELS),
        index=sorted(GPU_MODELS).index('HD Graphics 620'),
        help="Choose your graphics card"
    )
    
    st.divider()
    
    # Storage & Build Section
    st.write("### 💾 Storage & Build")
    
    memory = st.selectbox(
        "Storage Configuration", 
        options=sorted(MEMORY_OPTIONS),
        index=sorted(MEMORY_OPTIONS).index('512GB SSD'),
        help="SSD is faster than HDD"
    )
    
    weight = st.slider(
        "⚖️ Weight (KG)", 
        min_value=0.8, 
        max_value=5.0, 
        value=2.0, 
        step=0.1,
        help="Lighter laptops are more portable"
    )
    
    st.divider()
    st.info("💡 **Pro Tip:** Premium specs = Higher price!")

# ========== MAIN PAGE HEADER ==========
st.title("💻 Laptop Price Predictor")
st.write("### Predict laptop prices instantly using AI-powered machine learning")
st.write("")

# ========== CURRENT SPECIFICATIONS DISPLAY ==========
st.subheader("📋 Your Custom Configuration")

# Create 4 columns for specs display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🖥️ Screen", 
        value=f'{inches}"',
        delta=f"{screen_w}×{screen_h}",
        help="Screen size and resolution"
    )

with col2:
    st.metric(
        label="🧠 Memory", 
        value=f"{ram} GB",
        delta="RAM",
        help="Random Access Memory"
    )

with col3:
    # Extract CPU brand/series for display
    cpu_display = cpu.split()[0] if len(cpu.split()) > 0 else cpu
    st.metric(
        label="⚡ Processor", 
        value=cpu_display,
        delta="CPU",
        help=f"Full model: {cpu}"
    )

with col4:
    st.metric(
        label="💾 Storage", 
        value=memory.split()[0],
        delta=memory.split()[1] if len(memory.split()) > 1 else "",
        help="Storage capacity and type"
    )

# Additional specs in second row
col5, col6, col7, col8 = st.columns(4)

with col5:
    # Extract GPU brand for display
    gpu_display = gpu.split()[0] if len(gpu.split()) > 0 else gpu[:15]
    st.metric(
        label="🎮 Graphics", 
        value=gpu_display,
        delta="GPU",
        help=f"Full model: {gpu}"
    )

with col6:
    st.metric(
        label="⚖️ Weight", 
        value=f"{weight} kg",
        help="Device weight"
    )

with col7:
    st.metric(
        label="📐 Aspect", 
        value=f"{round(screen_w/screen_h, 2)}:1",
        help="Screen aspect ratio"
    )

with col8:
    ppi = round(np.sqrt(screen_w**2 + screen_h**2) / inches)
    st.metric(
        label="🔍 Clarity", 
        value=f"{ppi} PPI",
        help="Pixels per inch"
    )

st.divider()

# ========== PREDICTION SECTION ==========
st.subheader("🎯 Price Prediction")

# Center the button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🚀 Calculate Price Now", type="primary", use_container_width=True)

st.write("")

if predict_button:
    
    with st.spinner("🔮 AI is analyzing your configuration..."):
        # Create input dataframe with all features initialized to 0
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        
        # Fill numerical features
        input_data["Inches"] = inches
        input_data["Ram"] = ram
        input_data["Weight"] = weight
        input_data["ScreenW"] = screen_w
        input_data["ScreenH"] = screen_h
        
        # Fill categorical features (one-hot encoded)
        # CPU column
        cpu_col = f"CPU_{cpu}"
        if cpu_col in input_data.columns:
            input_data[cpu_col] = 1
        
        # GPU column
        gpu_col = f"GPU_{gpu}"
        if gpu_col in input_data.columns:
            input_data[gpu_col] = 1
        
        # Memory column
        mem_col = f"Memory_{memory}"
        if mem_col in input_data.columns:
            input_data[mem_col] = 1
        
        # Make prediction
        predicted_price_eur = model.predict(input_data)[0]
        predicted_price_inr = round(predicted_price_eur * 103.5)
    
    # Display success message
    st.success("✅ Prediction Complete!")
    st.balloons()
    
    # Display results in prominent cards
    st.write("")
    col_price1, col_price2 = st.columns(2)
    
    with col_price1:
        st.container(border=True)
        st.write("### 💶 European Market")
        st.metric(
            label="Price in Euros",
            value=f"€ {predicted_price_eur:,.2f}",
            delta="EUR",
            delta_color="off"
        )
        st.progress(min(predicted_price_eur / 3000, 1.0))
        if predicted_price_eur < 500:
            st.caption("🟢 Budget-friendly option")
        elif predicted_price_eur < 1500:
            st.caption("🟡 Mid-range laptop")
        else:
            st.caption("🔴 Premium/High-end device")
    
    with col_price2:
        st.container(border=True)
        st.write("### 💰 Indian Market")
        st.metric(
            label="Price in Rupees",
            value=f"₹ {predicted_price_inr:,}",
            delta="INR",
            delta_color="off"
        )
        st.progress(min(predicted_price_inr / 300000, 1.0))
        if predicted_price_inr < 50000:
            st.caption("🟢 Budget segment")
        elif predicted_price_inr < 150000:
            st.caption("🟡 Mid-range segment")
        else:
            st.caption("🔴 Premium segment")
    
    st.write("")
    
    # Price breakdown in expander
    with st.expander("📊 View Detailed Breakdown"):
        st.write("#### Price Calculation Details")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.write("**Euro Analysis:**")
            st.write(f"- Base Price: € {predicted_price_eur:.2f}")
            st.write(f"- Price per GB RAM: € {predicted_price_eur/ram:.2f}")
            st.write(f"- Price per inch: € {predicted_price_eur/inches:.2f}")
        
        with breakdown_col2:
            st.write("**Rupee Analysis:**")
            st.write(f"- Base Price: ₹ {predicted_price_inr:,}")
            st.write(f"- Conversion Rate: 1 EUR = 103.5 INR")
            st.write(f"- Price per GB RAM: ₹ {round(predicted_price_inr/ram):,}")
    
    # Specifications Summary
    with st.expander("📝 Configuration Summary"):
        st.write("#### Complete Specifications")
        
        spec_col1, spec_col2 = st.columns(2)
        
        with spec_col1:
            st.write("**Display:**")
            st.write(f"- Screen Size: {inches} inches")
            st.write(f"- Resolution: {screen_w} × {screen_h}")
            st.write(f"- PPI: {ppi}")
            st.write("")
            st.write("**Performance:**")
            st.write(f"- RAM: {ram} GB")
            st.write(f"- CPU: {cpu}")
        
        with spec_col2:
            st.write("**Graphics & Storage:**")
            st.write(f"- GPU: {gpu}")
            st.write(f"- Storage: {memory}")
            st.write("")
            st.write("**Physical:**")
            st.write(f"- Weight: {weight} kg")
    
    # Comparison section
    with st.expander("💡 Compare With Market Segments"):
        comparison_data = {
            "Segment": ["Budget", "Mid-Range", "Premium", "Your Laptop"],
            "EUR (€)": [300, 800, 1800, round(predicted_price_eur)],
            "INR (₹)": ["31,050", "82,800", "1,86,300", f"{predicted_price_inr:,}"]
        }
        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True
        )

st.divider()

# ========== MODEL PERFORMANCE SECTION ==========
st.subheader("📊 Model Performance Metrics")

r2_score = 0.8502661988677629
accuracy_percent = round(r2_score * 100, 2)

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric(
        label="🎯 Accuracy",
        value=f"{accuracy_percent}%",
        help="R² score of the model"
    )

with perf_col2:
    st.metric(
        label="🤖 Algorithm",
        value="Random Forest",
        help="Machine learning model used"
    )

with perf_col3:
    st.metric(
        label="📚 Features",
        value=f"{len(features)}",
        help="Number of features used in prediction"
    )

with perf_col4:
    st.metric(
        label="🔄 Model Version",
        value="v2.0",
        help="Current model version"
    )

st.progress(r2_score)
st.caption(f"Model Performance: {accuracy_percent}% accurate predictions based on R² score")

st.divider()

# ========== INFORMATION SECTION ==========
col_info1, col_info2 = st.columns(2)

with col_info1:
    with st.expander("ℹ️ How It Works"):
        st.write("""
        **Machine Learning Process:**
        
        1. **Data Collection**: Gathered comprehensive laptop specifications and prices
        2. **Feature Engineering**: Processed specs including:
           - {} CPU models
           - {} GPU models
           - Screen specifications
           - RAM and storage configurations
        3. **Model Training**: Used Random Forest algorithm for predictions
        4. **Validation**: Achieved {}% accuracy on test data
        5. **Deployment**: Ready to predict prices for any configuration
        
        **Key Features Analyzed:**
        - Screen size and resolution (PPI calculation)
        - RAM capacity (2GB to 64GB)
        - Processor models (Intel, AMD, Samsung)
        - Graphics cards (Intel, Nvidia, AMD, ARM)
        - Storage type and capacity
        - Device weight
        """.format(len(CPU_MODELS), len(GPU_MODELS), accuracy_percent))

with col_info2:
    with st.expander("⚠️ Important Notes"):
        st.write("""
        **Please Note:**
        
        - Prices are **estimates** based on specifications
        - Actual prices may vary by:
          - Brand (Dell, HP, Lenovo, Apple, Asus, etc.)
          - Market conditions and availability
          - Additional features (touchscreen, backlit keyboard)
          - Warranty and support packages
          - Regional taxes and duties
          - Retailer markup
        
        - This tool is for **reference only**
        - Always check current market prices before purchase
        - Consider checking multiple retailers
        - Look for seasonal sales and discounts
        
        **Supported Components:**
        - {} unique CPU models
        - {} unique GPU models
        - Multiple storage configurations
        """.format(len(CPU_MODELS), len(GPU_MODELS)))

st.divider()

# ========== FOOTER ==========
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.caption("Made with ❤️ using Streamlit | © 2024 Laptop Price Predictor")
    st.caption("🔒 Your data is processed locally and never stored")