
import requests
from datetime import datetime, timedelta
from config import DB_PATH
import sqlite3

def interval_to_milliseconds(interval):
    intervals = {'D': 86400000, '60': 3600000, '15': 900000, '5': 300000, '3': 180000, '1': 60000}
    return intervals.get(interval, 0)

def fetch_historical_data(symbol, interval, period):
    end_date = datetime.now()
    if period == '5 лет':
        start_date = end_date - timedelta(days=5 * 365)
    elif period == '3 года':
        start_date = end_date - timedelta(days=3 * 365)
    elif period == '1 год':
        start_date = end_date - timedelta(days=365)
    elif period == 'полгода':
        start_date = end_date - timedelta(days=180)
    elif period == '1 месяц':
        start_date = end_date - timedelta(days=30)
    elif period == '1 неделя':
        start_date = end_date - timedelta(days=7)
    elif period == '1 день':
        start_date = end_date - timedelta(days=1)
    else:
        start_date = end_date - timedelta(days=365)

    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    url = "https://api.bybit.com/v5/market/kline"
    all_data = []

    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval, "start": start_time, "limit": 1000}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode", -1) != 0:
                print(f"Ошибка API: {data.get('retMsg', 'Неизвестная ошибка')}")
                break
            candles = data["result"]["list"]
            if not candles:
                break
            for candle in candles:
                timestamp = int(candle[0]) // 1000
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])
                all_data.append((timestamp, open_price, high_price, low_price, close_price, volume))
            last_candle_time = int(candles[0][0])
            start_time = last_candle_time + interval_to_milliseconds(interval)
            if len(candles) < 1000:
                break
        except Exception as e:
            print(f"Ошибка при запросе к API: {e}")
            break

    all_data.sort(key=lambda x: x[0])
    return all_data
