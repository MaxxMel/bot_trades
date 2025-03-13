import sqlite3
from config import DB_PATH

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bitcoin_historical (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)
    conn.commit()
    conn.close()

def clear_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bitcoin_historical")
    conn.commit()
    conn.close()

def save_data_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO bitcoin_historical (timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM bitcoin_historical"
    import pandas as pd
    df_local = pd.read_sql_query(query, conn)
    conn.close()
    return df_local
