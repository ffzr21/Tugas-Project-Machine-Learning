import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

model = load_model()

# ============================================================
# Header
# ============================================================
st.title("💻 Laptop Price Predictor")
st.markdown(
    """
    Aplikasi ini memprediksi **estimasi harga laptop** berdasarkan spesifikasinya
    menggunakan model machine learning yang telah dilatih (lihat notebook
    `laporan_submission_laptop_price.ipynb` untuk detail proses CRISP-DM).
    """
)

st.divider()

# ============================================================
# Form Input
# ============================================================
st.subheader("Masukkan Spesifikasi Laptop")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox(
        "Merek (Company)",
        ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Apple', 'Samsung', 'Razer']
    )

    type_name = st.selectbox(
        "Tipe Laptop",
        ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook']
    )

    ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)

    weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    cpu_brand = st.selectbox(
        "Brand CPU",
        ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9',
         'AMD Processor', 'Other Intel Processor']
    )

    cpu_speed = st.number_input("Kecepatan CPU (GHz)", min_value=1.0, max_value=4.0, value=2.5, step=0.1)

with col2:
    screen_size = st.number_input("Ukuran Layar (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)

    resolution = st.selectbox(
        "Resolusi Layar",
        ['1366x768', '1920x1080', '2560x1440', '3840x2160']
    )

    touchscreen = st.selectbox("Touchscreen?", ['No', 'Yes'])
    ips = st.selectbox("IPS Panel?", ['No', 'Yes'])

    hdd = st.selectbox("Kapasitas HDD (GB)", [0, 128, 256, 500, 1024, 2048], index=0)
    ssd = st.selectbox("Kapasitas SSD (GB)", [0, 128, 256, 512, 1024], index=2)

    gpu_brand = st.selectbox("Brand GPU", ['Intel', 'Nvidia', 'AMD'])

    os_choice = st.selectbox("Sistem Operasi", ['Windows', 'Mac', 'Other', 'No OS'])

st.divider()

# ============================================================
# Prediksi
# ============================================================
if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):

    # Hitung PPI dari resolusi
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': 1 if touchscreen == 'Yes' else 0,
        'IPS': 1 if ips == 'Yes' else 0,
        'PPI': ppi,
        'Cpu_brand': cpu_brand,
        'Cpu_speed': cpu_speed,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu_brand': gpu_brand,
        'os': os_choice
    }])

    pred_log = model.predict(input_df)[0]
    pred_price = np.exp(pred_log)

    EURO_TO_IDR = 17500  # kurs bisa disesuaikan
    pred_price_idr = pred_price * EURO_TO_IDR
    st.success(f"### 💰 Estimasi Harga: Rp {pred_price_idr:,.0f}")
    st.caption(f"(Estimasi dalam Euro: €{pred_price:,.2f} × kurs Rp {EURO_TO_IDR:,})")

    with st.expander("Lihat data input yang digunakan model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Catatan: Harga merupakan estimasi yang dihasilkan oleh model machine learning "
    "berdasarkan pola data latih, dan dapat berbeda dengan harga pasar aktual."
)
