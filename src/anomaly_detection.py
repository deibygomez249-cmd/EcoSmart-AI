import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.data_processing import load_and_prepare_data

def detect_anomalies(contamination=0.05):
    df, _, _, _, _, _ = load_and_prepare_data()
    
    model = IsolationForest(contamination=contamination, random_state=42)
    
    X_features = df[['T1', 'T2', 'T3', 'T4', 'T5', 'RH_1', 'RH_2', 'RH_3']]
    
    df['Anomaly'] = model.fit_predict(X_features)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    anomalies_count = (df['Anomaly'] == 'Anomaly').sum()
    total = len(df)
    percentage = round((anomalies_count / total) * 100, 2)
    
    anomaly_records = df[df['Anomaly'] == 'Anomaly'][['date', 'Appliances', 'Anomaly']].tail(50)
    
    return {
        'total': total,
        'anomalies_count': anomalies_count,
        'percentage': percentage,
        'records': anomaly_records.to_dict(orient='records')
    }