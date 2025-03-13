import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def send_plot(chat_id, plot_func, bot):
    import io
    buf = io.BytesIO()
    plot_func()
    plt.savefig(buf, format='png')
    buf.seek(0)
    bot.send_photo(chat_id, buf)
    plt.close()

def plot_correlation_matrix_matplotlib(df_local, feats):
    if not feats:
        plt.title("Нет доступных факторов для корреляции")
        return
    corr_matrix = df_local[feats].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90)
    ax.set_yticklabels(feats)
    plt.title("Correlation Matrix", pad=20)

def plot_price(df_local):
    plt.figure()
    plt.plot(df_local['timestamp'], df_local['close'], label='Close Price')
    plt.title("Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

def plot_factors(df_local, feats):
    if df_local is None or df_local.empty or not feats:
        return
    plt.figure()
    plt.plot(df_local['timestamp'], df_local['close'], label='Close Price')
    for f in feats:
        plt.plot(df_local['timestamp'], df_local[f], label=f, linestyle='--')
    plt.title("Price and Technical Indicators")
    plt.legend()

def plot_predictions(df_local, X_test, y_test, y_pred, model_name):
    test_index = X_test.index
    df_test = df_local.loc[test_index].copy()
    df_test['y_true'] = y_test
    df_test['y_pred'] = y_pred
    plt.figure()
    plt.plot(df_test['timestamp'], df_test['close'], label='Close Price')
    long_points = df_test[df_test['y_pred'] == 1]
    short_points = df_test[df_test['y_pred'] == 0]
    plt.scatter(long_points['timestamp'], long_points['close'], marker='^', label='Long (pred=1)')
    plt.scatter(short_points['timestamp'], short_points['close'], marker='v', label='Short (pred=0)')
    plt.title(f"Model Predictions: {model_name}")
    plt.legend()

def plot_capital_curve(df_local, capital_series):
    plt.figure()
    plt.plot(capital_series.index, capital_series.values)
    plt.title("Capital Curve Over Test Period")
    plt.xlabel("Date Index")
    plt.ylabel("Capital")
