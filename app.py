import streamlit as st
import pandas as pd
from src.inference import PUBG_Predictor

# Sayfa Ayarları
st.set_page_config(page_title="PUBG Win Predictor", page_icon="🎮", layout="centered")

# Başlık ve Açıklama
st.title("🎮 PUBG Win Probability Predictor")
st.markdown("""
This project uses **Machine Learning** to predict your winning probability (**WinPlacePerc**) 
based on your in-game statistics.
""")

st.image("banner.jpg", caption="Winner Winner Chicken Dinner!", use_column_width=True)

# Sidebar - Model Bilgisi
st.sidebar.header("About the Project")
st.sidebar.info("This model was trained using LightGBM on over 4.4 Million match records from the official PUBG dataset.")
st.sidebar.markdown("---")
st.sidebar.write("Developer: whitephysics")

# ----- KULLANICI GİRİŞLERİ -----
st.subheader("📊 Enter Match Statistics")

col1, col2 = st.columns(2)

with col1:
    walkDistance = st.number_input("Walk Distance (m)", min_value=0, value=1000, step=100)
    kills = st.number_input("Kills", min_value=0, value=2, step=1)
    damageDealt = st.number_input("Damage Dealt", min_value=0.0, value=200.0, step=50.0)

with col2:
    boosts = st.number_input("Boosts (Energy Drink/Painkillers)", min_value=0, value=1, step=1)
    heals = st.number_input("Heals (Bandages/First Aid)", min_value=0, value=1, step=1)
    matchType = st.selectbox("Game Mode", ["squad-fpp", "duo-fpp", "solo-fpp", "squad", "duo", "solo"])

# Diğer detaylar (Varsayılan veya basit girişler)
revives = 0 # Basitlik adına arayüze koymadık, kodda 0 gidecek veya eklenebilir
assists = 0

# Tahmin Butonu
if st.button("🚀 Predict Win Probability"):
    # Veriyi DataFrame'e çevir
    input_data = pd.DataFrame({
        'walkDistance': [walkDistance],
        'rideDistance': [0], # Basitlik için 0 varsaydık
        'swimDistance': [0],
        'kills': [kills],
        'damageDealt': [damageDealt],
        'boosts': [boosts],
        'heals': [heals],
        'headshotKills': [0], # Detay sormadık
        'revives': [revives],
        'assists': [assists],
        'matchType': [matchType]
    })

    # Modeli Yükle ve Tahmin Et
    try:
        predictor = PUBG_Predictor()
        prediction = predictor.predict(input_data)[0]
        
        # Sonucu Göster
        st.divider()
        st.subheader("Result:")
        
        if prediction > 0.8:
            st.success(f"🏆 Winner Winner Chicken Dinner! Win Probability: %{prediction*100:.2f}")
            st.balloons()
        elif prediction > 0.5:
            st.warning(f"🤔 Top 10 Potential. Win Probability: %{prediction*100:.2f}")
        else:
            st.error(f"💀 Better Luck Next Time. Win Probability: %{prediction*100:.2f}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Error 'models/pubg_model.pkl' not found.")