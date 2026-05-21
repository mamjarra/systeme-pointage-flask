from datetime import datetime

def mark_attendance(conn):
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM presence WHERE date LIKE ?",
                   (f"{today}%",))
    nb  = cursor.fetchone()[0]
    nom = f"Personne_{nb + 1}"
    cursor.execute("INSERT INTO presence (nom, date) VALUES (?, ?)",
                   (nom, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return nom

def get_present_today(conn):
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT nom, date FROM presence WHERE date LIKE ? ORDER BY date DESC",
        (f"{today}%",))
    return cursor.fetchall()

def get_total_today(conn):
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM presence WHERE date LIKE ?",
                   (f"{today}%",))
    return cursor.fetchone()[0]