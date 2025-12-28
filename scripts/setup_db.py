import sqlite3
import os

def create_dummy_db(db_path="data/dummy_database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabelle erstellen
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER,
        hire_date DATE
    )
    """)

    # Daten einfügen
    employees = [
        (1, 'Alice Smith', 'Sales', 55000, '2021-01-15'),
        (2, 'Bob Jones', 'Engineering', 85000, '2020-03-10'),
        (3, 'Charlie Brown', 'Sales', 48000, '2022-06-23'),
        (4, 'Diana Prince', 'Engineering', 92000, '2019-11-05'),
        (5, 'Evan Wright', 'HR', 45000, '2021-09-30')
    ]
    
    cursor.executemany('INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?)', employees)
    conn.commit()
    conn.close()
    print(f"✅ Datenbank erstellt: {db_path}")

if __name__ == "__main__":
    create_dummy_db()