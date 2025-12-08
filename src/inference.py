import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pubg_model.pkl')

class PUBG_Predictor:
    def __init__(self, model_path=MODEL_PATH):
        
        "Modeli yükler ve kullanıma hazırlar."
        
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        
        "Eğitilmiş .pkl dosyasını yükler."
    
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✅ Model Successfully Loaded: {self.model_path}")
        else:
            raise FileNotFoundError(f"❌ Model File Not Found: {self.model_path}")

    def preprocess_data(self, df):
        "Ham veriyi alır, Feature Engineering işlemlerini uygular ve modelin beklediği formata getirir"

        # Veri kopyası üzerinde çalışalım
        data = df.copy()

        # ------------------- FEATURE ENGINEERING -------------------
        
        # 1. Basit Türetmeler
        # Eğer bu sütunlar yoksa 0 kabul et veya hesapla.
        if 'walkDistance' in data.columns and 'rideDistance' in data.columns:
            data['totalDistance'] = data['walkDistance'] + data['rideDistance'] + data.get('swimDistance', 0)
        
        if 'heals' in data.columns and 'boosts' in data.columns:
            data['healthItems'] = data['heals'] + data['boosts']
        
        if 'headshotKills' in data.columns and 'kills' in data.columns:
            data['headshotRate'] = data['headshotKills'] / data['kills']
            data['headshotRate'] = data['headshotRate'].fillna(0)
            
        if 'revives' in data.columns and 'assists' in data.columns:
            data['teamwork'] = data['revives'] + data['assists']

        # 2. Aggregations
        
        if 'matchId' in data.columns and len(data) > 1:
            # Batch Prediction
            data['matchMeanKills'] = data.groupby('matchId')['kills'].transform('mean')
            data['matchMeanDamage'] = data.groupby('matchId')['damageDealt'].transform('mean')
            
            data['killsRel'] = data['kills'] / data['matchMeanKills']
            data['damageRel'] = data['damageDealt'] / data['matchMeanDamage']
        else:
            data['killsRel'] = 0
            data['damageRel'] = 0
        
        # NaN temizliği.
        data['killsRel'] = data['killsRel'].fillna(0)
        data['damageRel'] = data['damageRel'].fillna(0)

        # Gereksiz sütunları at.
        drop_cols = ['Id', 'groupId', 'matchId', 'winPlacePerc']
        data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

        # Modelin beklediği feature listesini alalım.
        expected_features = self.model.booster_.feature_name()
        
        # Kategorik 'matchType' verisini işle.
        if 'matchType' in data.columns:
            data = pd.get_dummies(data, columns=['matchType'])
        
        # Eksik sütunları 0 ile doldur.
        for col in expected_features:
            if col not in data.columns:
                data[col] = 0
                
        # Sütun sırasını modelle aynı yap.
        data = data[expected_features]
        
        return data

    def predict(self, df):
        "Dışarıdan gelen veri için tahmin üretir."
        
        processed_data = self.preprocess_data(df)
        predictions = self.model.predict(processed_data)
        
        # Sonuçları 0-1 arasına sıkıştır.
        predictions = np.clip(predictions, 0, 1)
        
        return predictions

# # ------------------- TEST -------------------
if __name__ == "__main__":
    predictor = PUBG_Predictor()
    print("Generating predictions on dummy data...")
    
    # Sahte bir veri oluşturalım.
    dummy_data = pd.DataFrame({
        'matchId': ['m1', 'm1'], # 2 kişilik bir maç simülasyonu
        'groupId': ['g1', 'g2'],
        'kills': [5, 2],
        'damageDealt': [400, 100],
        'walkDistance': [2000, 500],
        'rideDistance': [1000, 0],
        'swimDistance': [0, 0],
        'heals': [2, 0],
        'boosts': [3, 0],
        'headshotKills': [1, 0],
        'revives': [1, 0],
        'assists': [0, 0],
        'matchType': ['squad', 'squad']
    })
    
    sonuc = predictor.predict(dummy_data)

    print(f"Pred Results: {sonuc}")
