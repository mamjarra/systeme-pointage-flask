import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pointage.db")

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db()
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS presence (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            nom  TEXT NOT NULL,
            date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()