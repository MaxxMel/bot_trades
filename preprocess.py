import pandas as pd
from datetime import datetime

def preprocess_data(df_local, fill_method=None):
    if df_local is None or df_local.empty:
        return None
    df_local['timestamp'] = pd.to_datetime(df_local['timestamp'], unit='s')
    if fill_method in ['ffill', 'bfill']:
        df_local.fillna(method=fill_method, inplace=True)
    else:
        df_local.dropna(inplace=True)
    return df_local

def create_binary_target(df_local, target_col='close', shift_n=1):
    if df_local is None or df_local.empty:
        return None
    df_local['future_price'] = df_local[target_col].shift(-shift_n)
    df_local.dropna(subset=['future_price'], inplace=True)
    df_local['target'] = (df_local['future_price'] > df_local[target_col]).astype(int)
    return df_local

def remove_correlated_features(df_local, features, threshold=0.8):
    if df_local is None or df_local.empty or not features:
        return [], features
    corr_matrix = df_local[features].corr().abs()
    upper = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    remaining = [f for f in features if f not in to_drop]
    return to_drop, remaining
