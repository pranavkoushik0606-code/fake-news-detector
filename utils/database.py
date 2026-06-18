import sqlite3

def create_table():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news TEXT,
        prediction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(news, prediction):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO predictions (news, prediction) VALUES (?, ?)",
        (news, prediction)
    )

    conn.commit()
    conn.close()


def get_predictions():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute(
        "SELECT * FROM predictions ORDER BY timestamp DESC"
    )

    data = c.fetchall()

    conn.close()

    return data