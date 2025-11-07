import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi CKD (Decision Tree) - Final",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul dan Deskripsi
st.title("🌳 Aplikasi Prediksi Penyakit Ginjal Kronis")
st.markdown("""
Aplikasi Decision Tree ini telah dioptimalkan untuk mengatasi peringatan tipe data Streamlit. 
Semua input dibatasi oleh rentang nilai data pelatihan dan memiliki format desimal yang sesuai.
""")
st.divider()

# --- Fungsi Pemuatan Data dan Pelatihan Model ---

# Metadata format khusus untuk tampilan dan langkah input
CUSTOM_FORMATS = {
    # FIX: Mengubah %d menjadi %.0f untuk fitur integer agar kompatibel dengan value=float
    'age': ('%.0f', 1.0),
    'bp': ('%.0f', 1.0),
    'bu': ('%.0f', 1.0),
    # Format lain tetap
    'al': ('%.1f', 0.1),
    'su': ('%.1f', 0.1),
    'sg': ('%.3f', 0.001), # Specific Gravity: 3 desimal
    'bgr': ('%.1f', 0.1),
    'sc': ('%.1f', 0.1),
    'hemo': ('%.1f', 0.1),
}

@st.cache_data
def load_data(file_path):
    """Memuat data dan menghitung metadata rentang."""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna() 
        
        feature_names = df.columns.drop('classification').tolist()
        X = df[feature_names]
        y = df['classification']
        
        # Hitung metadata (min, max, mean) untuk input rentang
        metadata = {}
        for col in feature_names:
            metadata[col] = {
                'min': X[col].min(),
                'max': X[col].max(),
                'mean': X[col].mean(),
                'format': CUSTOM_FORMATS[col][0],
                'step': CUSTOM_FORMATS[col][1]
            }
        
        return X, y, feature_names, metadata
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan: {file_path}. Pastikan 'cleaned_data.csv' ada.")
        return None, None, None, None

@st.cache_resource
def train_model(X, y):
    """Melatih model Decision Tree dan Standard Scaler."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    X_test_scaled = scaler.transform(X_test)
    accuracy = model.score(X_test_scaled, y_test)
    
    return model, scaler, accuracy

# --- Pemuatan Data dan Pelatihan ---
X, y, feature_names, metadata = load_data('cleaned_data.csv')

if X is not None:
    model, scaler, accuracy = train_model(X, y)
    
    # --- Sidebar untuk Input Pengguna ---
    st.sidebar.success(f"Model Decision Tree Dilatih. Akurasi Uji: **{accuracy:.2f}**")
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Masukkan Data Pasien")
    
    
    # Pembagian Input menjadi dua kolom di sidebar
    col_input1, col_input2 = st.sidebar.columns(2)
    
    
    # Fungsi untuk membuat st.number_input dengan batasan dan format kustom
    def create_bounded_number_input(col, label, container):
        meta = metadata[col]
        
        # Pastikan nilai default (mean) berada dalam batas min/max
        default_value = np.clip(meta['mean'], meta['min'], meta['max'])
        
        return container.number_input(
            label=f"{label} ({col})",
            # min_value dan max_value diubah ke float (untuk menghindari error sebelumnya)
            min_value=float(meta['min']),
            max_value=float(meta['max']),
            # value, step, dan format disesuaikan
            value=float(default_value),
            step=meta['step'],
            format=meta['format'],
            key=col
        )
    
    # --- Input Data ke Kolom 1 (Sidebar) ---
    with col_input1:
        st.subheader("Demografi & Fisik")
        age = create_bounded_number_input('age', "1. Usia (Tahun)", col_input1)
        bp = create_bounded_number_input('bp', "2. Tekanan Darah (mm/Hg)", col_input1)
        sg = create_bounded_number_input('sg', "3. Specific Gravity", col_input1)
        al = create_bounded_number_input('al', "4. Albumin", col_input1)
        su = create_bounded_number_input('su', "5. Sugar", col_input1)

    # --- Input Data ke Kolom 2 (Sidebar) ---
    with col_input2:
        st.subheader("Hasil Lab")
        bgr = create_bounded_number_input('bgr', "6. Glukosa Darah Acak", col_input2)
        bu = create_bounded_number_input('bu', "7. Blood Urea", col_input2)
        sc = create_bounded_number_input('sc', "8. Serum Creatinine", col_input2)
        hemo = create_bounded_number_input('hemo', "9. Hemoglobin", col_input2)
        
        st.write("") 

    # --- Pengumpulan Data Input untuk Prediksi ---
    user_data = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'bgr': bgr, 'bu': bu, 'sc': sc, 'hemo': hemo
    }
    
    input_df = pd.DataFrame([user_data])
    input_df = input_df[feature_names] # Pastikan urutan kolom

    # --- Tampilan Data Input (Main Content) ---
    st.subheader("Data Klinis yang Diinputkan")
    st.dataframe(input_df.style.set_properties(**{'background-color': '#e0f7fa', 'color': 'black'}), use_container_width=True)
    st.divider()

    # --- Tombol dan Hasil Prediksi ---
    
    if st.button("🚀 PREDIKSI", type="primary"):
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        ckd_proba = prediction_proba[0][1] * 100
        
        st.subheader("✅ Hasil Klasifikasi")
        
        col_res1, col_res2 = st.columns([1, 3])
        
        with col_res1:
            st.metric(label="Probabilitas Penyakit Ginjal Kronis (CKD)", value=f"{ckd_proba:.2f}%")
            
        with col_res2:
            if prediction[0] == 1.0:
                st.error("Status: **RISIKO TINGGI PENYAKIT GINJAL KRONIS**", icon="🚨")
                st.markdown("""
                > **Perhatian:** Hasil menunjukkan prediksi CKD. Diperlukan konfirmasi dan pemeriksaan lanjutan oleh dokter spesialis.
                """)
                st.balloons()
            else:
                st.success("Status: **RISIKO RENDAH PENYAKIT GINJAL KRONIS**", icon="✅")
                st.markdown("""
                > **Baik:** Prediksi Non-CKD. Tetap jaga kesehatan Anda dan lakukan pemeriksa an rutin.
                """) 