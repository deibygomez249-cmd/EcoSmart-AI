import os
import io
import json
import time
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join('data', 'energydata_complete.csv')
ARTIFACTS_DIR = 'artifacts'
METADATA_PATH = os.path.join(ARTIFACTS_DIR, 'ecosmart_metadata.json')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model_metadata = {
    'regression': {'results': {}, 'best': ''},
    'classification': {'results': {}, 'threshold': 0},
    'trained_at': ''
}
trained_models = {}

def _fig_to_base64(fig):
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
    img.seek(0)
    b64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return b64

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found. Place 'energydata_complete.csv' in 'data/'.")
    
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    df = df.dropna()
    df = df.drop(columns=[c for c in ['lights', 'rv1', 'rv2'] if c in df.columns])
    
    target = 'Appliances'
    X = df.drop(columns=[target, 'date'], errors='ignore')
    y = df[target]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return df, X_train, X_test, y_train, y_test

def train_all_models():
    global trained_models, model_metadata
    df, X_train, X_test, y_train, y_test = load_data()
    
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    reg_results = {}
    for name, model in reg_models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        reg_results[name] = {
            'mae': round(mean_absolute_error(y_test, preds), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            'r2': round(r2_score(y_test, preds), 3),
            'time': round(time.time() - start, 2)
        }
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"reg_{name.replace(' ', '_').lower()}.joblib"))
        trained_models[f"reg_{name}"] = model
        
    best_reg = min(reg_results, key=lambda k: reg_results[k]['rmse'])

    threshold = y_train.quantile(0.75)
    y_train_class = (y_train > threshold).astype(int)
    y_test_class = (y_test > threshold).astype(int)
    
    clf_models = {
        'Ridge Classifier': RidgeClassifier(),
        'Random Forest Clf': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }
    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_train_class)
        preds = model.predict(X_test)
        clf_results[name] = {
            'accuracy': round(accuracy_score(y_test_class, preds), 3),
            'f1': round(f1_score(y_test_class, preds), 3)
        }
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"clf_{name.replace(' ', '_').lower()}.joblib"))
        trained_models[f"clf_{name}"] = model

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)
    trained_models['iso_forest'] = iso
    joblib.dump(iso, os.path.join(ARTIFACTS_DIR, 'iso_forest.joblib'))

    model_metadata = {
        'regression': {'results': reg_results, 'best': best_reg},
        'classification': {'results': clf_results, 'threshold': threshold},
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(model_metadata, f)
        
    return model_metadata

def load_saved_models():
    global trained_models, model_metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            loaded_data = json.load(f)
            if 'regression' in loaded_data:
                model_metadata = loaded_data
        
        for name in ['Linear_Regression', 'Random_Forest', 'HistGradientBoosting']:
            path = os.path.join(ARTIFACTS_DIR, f"reg_{name.lower()}.joblib")
            if os.path.exists(path):
                trained_models[f"reg_{name.replace('_', ' ')}"] = joblib.load(path)
                
        for name in ['Ridge_Classifier', 'Random_Forest_Clf']:
            path = os.path.join(ARTIFACTS_DIR, f"clf_{name.lower()}.joblib")
            if os.path.exists(path):
                trained_models[f"clf_{name.replace('_', ' ')}"] = joblib.load(path)
                
        path_iso = os.path.join(ARTIFACTS_DIR, 'iso_forest.joblib')
        if os.path.exists(path_iso):
            trained_models['iso_forest'] = joblib.load(path_iso)

def get_dashboard_charts():
    df, _, _, _, _ = load_data()
    
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    fig1.patch.set_facecolor('#1e1e2f')
    ax1.set_facecolor('#1e1e2f')
    sample = df.head(1000)
    ax1.plot(sample['date'], sample['Appliances'], color='#00e676', linewidth=1.2)
    ax1.fill_between(sample['date'], sample['Appliances'], color='#00e676', alpha=0.15)
    ax1.set_title('Energy Timeline (First 1000 records)', color='white', fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, color='#444', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_timeline = _fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor('#1e1e2f')
    ax2.set_facecolor('#1e1e2f')
    ax2.hist(df['Appliances'], bins=50, color='#4facfe', alpha=0.8)
    ax2.set_title('Consumption Distribution', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, color='#444', alpha=0.3)
    plt.tight_layout()
    plot_dist = _fig_to_base64(fig2)

    return {
        'records': len(df),
        'avg': round(df['Appliances'].mean(), 2),
        'max': round(df['Appliances'].max(), 2),
        'min': round(df['Appliances'].min(), 2),
        'plot_timeline': plot_timeline,
        'plot_dist': plot_dist
    }

def get_clustering_plot():
    df, _, _, _, _ = load_data()
    sample = df[['Appliances', 'T_out']].sample(n=min(2000, len(df)), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    colors = ['#ff4b4b', '#00e676', '#4facfe']
    for i in range(3):
        cluster_data = sample[labels == i]
        ax.scatter(cluster_data['T_out'], cluster_data['Appliances'], 
                   c=colors[i], label=f'Profile {i+1}', alpha=0.6, edgecolors='white')
        
    ax.set_title('K-Means: Usage Profiles', color='white', fontweight='bold')
    ax.set_xlabel('Outdoor Temperature (°C)', color='white')
    ax.set_ylabel('Consumption (Wh)', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.grid(True, color='#444', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)

def get_anomaly_plot():
    df, _, _, _, _ = load_data()
    iso = trained_models.get('iso_forest')
    if not iso: return None
    
    X_full = df.drop(columns=['Appliances', 'date'], errors='ignore')
    anomalies = iso.predict(X_full)
    df['Anomaly'] = pd.Series(anomalies).map({1: 'Normal', -1: 'Anomaly'})

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    sample = df.head(1000)
    normal = sample[sample['Anomaly'] == 'Normal']
    anom = sample[sample['Anomaly'] == 'Anomaly']
    
    ax.scatter(normal['date'], normal['Appliances'], color='#4facfe', s=10, label='Normal')
    ax.scatter(anom['date'], anom['Appliances'], color='#ff4b4b', s=30, marker='x', label='Anomaly')
    
    ax.set_title('Isolation Forest: Anomaly Detection', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, color='#444', alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64(fig)

def get_metrics_chart():
    if not model_metadata or 'regression' not in model_metadata or not model_metadata['regression']['results']: return None
    results = model_metadata['regression']['results']
    models = list(results.keys())
    rmses = [results[m]['rmse'] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    bars = ax.bar(models, rmses, color=['#6c757d', '#00e676', '#4facfe'])
    ax.set_title('Model Comparison (RMSE)', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', color='white')
        
    plt.tight_layout()
    return _fig_to_base64(fig)

def predict_single(features_dict):
    if not trained_models or 'regression' not in model_metadata: raise ValueError("Train models first.")
    best_name = model_metadata['regression']['best']
    model = trained_models.get(f"reg_{best_name}")
    if not model: raise ValueError("Best model not found in memory.")
    
    _, X_train, _, _, _ = load_data()
    input_df = pd.DataFrame([features_dict])[X_train.columns]
    pred = model.predict(input_df)[0]
    return round(float(pred), 2)

def get_zone_data():
    df, _, _, _, _ = load_data()
    zones = [
        {'name': 'Kitchen', 'temp': round(df['T1'].mean(), 1), 'hum': round(df['RH_1'].mean(), 1)},
        {'name': 'Living room', 'temp': round(df['T2'].mean(), 1), 'hum': round(df['RH_2'].mean(), 1)},
        {'name': 'Laundry', 'temp': round(df['T3'].mean(), 1), 'hum': round(df['RH_3'].mean(), 1)},
        {'name': 'Office', 'temp': round(df['T4'].mean(), 1), 'hum': round(df['RH_4'].mean(), 1)},
        {'name': 'Outdoor', 'temp': round(df['T_out'].mean(), 1), 'hum': round(df['RH_out'].mean(), 1), 'is_outdoor': True}
    ]
    return zones

def get_data_explorer_info():
    df, _, _, _, _ = load_data()
    shape = df.shape
    missing = df.isnull().sum().sum()
    
    cols_to_show = ['date', 'Appliances', 'T1', 'RH_1', 'T_out', 'RH_out', 'Windspeed', 'Press_mm_hg']
    available_cols = [c for c in cols_to_show if c in df.columns]
    
    head_html = df[available_cols].head(15).to_html(classes='table table-dark-custom table-sm align-middle text-nowrap', index=False, border=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    cols_corr = ['Appliances', 'T1', 'T2', 'T3', 'T_out', 'RH_1', 'RH_out', 'Windspeed']
    available_corr = [c for c in cols_corr if c in df.columns]
    corr = df[available_corr].corr()
    
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix', color='white', fontweight='bold', pad=15)
    ax.tick_params(colors='white')
    plt.tight_layout()
    corr_plot = _fig_to_base64(fig)
    
    return {
        'shape': shape,
        'missing': missing,
        'head_html': head_html,
        'corr_plot': corr_plot
    }

def get_anomalies_data(contamination=0.05):
    df, _, _, _, _ = load_data()
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    X = df.drop(columns=['Appliances', 'date'], errors='ignore')
    anomalies = iso.fit_predict(X)
    
    df['Anomaly'] = pd.Series(anomalies).map({1: 'Normal', -1: 'Anomaly'})
    
    anom_df = df[df['Anomaly'] == 'Anomaly']
    normal_df = df[df['Anomaly'] == 'Normal']
    
    total_anom = len(anom_df)
    percentage = round((total_anom / len(df)) * 100, 2)
    
    avg_anom_cons = round(anom_df['Appliances'].mean(), 2) if total_anom > 0 else 0
    avg_normal_cons = round(normal_df['Appliances'].mean(), 2)
    
    max_anom_record = anom_df.loc[anom_df['Appliances'].idxmax()] if total_anom > 0 else None
    max_anom_val = round(float(max_anom_record['Appliances']), 2) if max_anom_record is not None else 0
    max_anom_date = max_anom_record['date'].strftime('%Y-%m-%d %H:%M') if max_anom_record is not None else 'N/A'

    cols_show = ['date', 'Appliances', 'T_out', 'RH_out', 'Windspeed']
    anom_table = anom_df[cols_show].head(15).to_html(classes='table table-dark-custom table-sm align-middle', index=False, border=0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    sample = df.head(1000)
    normal_sample = sample[sample['Anomaly'] == 'Normal']
    anom_sample = sample[sample['Anomaly'] == 'Anomaly']
    
    ax.scatter(normal_sample['date'], normal_sample['Appliances'], color='#4facfe', s=10, label='Normal', alpha=0.6)
    ax.scatter(anom_sample['date'], anom_sample['Appliances'], color='#ff4b4b', s=40, marker='X', label='Anomaly')
    
    ax.set_title('Anomaly Detection Timeline', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, color='#444', alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_url = _fig_to_base64(fig)
    
    return {
        'total_anom': total_anom,
        'percentage': percentage,
        'avg_anom_cons': avg_anom_cons,
        'avg_normal_cons': avg_normal_cons,
        'max_anom_val': max_anom_val,
        'max_anom_date': max_anom_date,
        'table': anom_table,
        'plot_url': plot_url
    }

def get_insights_data():
    df, _, _, _, _ = load_data()
    
    df['hour'] = df['date'].dt.hour
    peak_hour = int(df.groupby('hour')['Appliances'].mean().idxmax())
    
    df['day_name'] = df['date'].dt.day_name()
    peak_day = df.groupby('day_name')['Appliances'].mean().idxmax()
    
    if df['T_out'].corr(df['Appliances']) < 0:
        weather_insight = "There is a negative correlation with outdoor temperature. Consumption tends to be higher when it is colder outside (likely due to heating systems)."
    else:
        weather_insight = "There is a positive correlation with outdoor temperature (likely due to cooling systems)."
        
    day_avg = df[df['hour'].between(8, 18)]['Appliances'].mean()
    night_avg = df[~df['hour'].between(8, 18)]['Appliances'].mean()
    
    return {
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'weather_insight': weather_insight,
        'day_avg': round(day_avg, 2),
        'night_avg': round(night_avg, 2)
    }

try:
    load_saved_models()
except Exception as e:
    print("Error loading models:", e)