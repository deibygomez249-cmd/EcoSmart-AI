import os
import pandas as pd

DATA_PATH = os.path.join('data', 'energydata_complete.csv')

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Please download 'energydata_complete.csv' from UCI "
            "and place it in the 'data/' folder."
        )
    
    df = pd.read_csv(DATA_PATH)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    df = df.dropna()
    
    target_col = 'Appliances'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    features = ['T1', 'T2', 'T3', 'T4', 'T5', 'RH_1', 'RH_2', 'RH_3']
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df[target_col]
    
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    return df, X_train, X_test, y_train, y_test, available_features

def get_dataset_info():
    df = pd.read_csv(DATA_PATH)
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': int(df.isnull().sum().sum()),
        'preview': df.head(10).to_html(classes='table table-striped table-hover', index=False)
    }
