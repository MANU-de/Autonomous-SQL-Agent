import gradio as gr
import sqlite3
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- TEIL 1: Die Dummy-Datenbank ---
DB_PATH = "dummy_database.db"

def setup_db():
    """Erstellt die Datenbank bei jedem Neustart frisch (ideal für Demos)"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH) # Aufräumen für sauberen Start
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER,
        hire_date DATE
    )
    """)
    employees = [
        (1, 'Alice Smith', 'Sales', 55000, '2021-01-15'),
        (2, 'Bob Jones', 'Engineering', 85000, '2020-03-10'),
        (3, 'Charlie Brown', 'Sales', 48000, '2022-06-23'),
        (4, 'Diana Prince', 'Engineering', 92000, '2019-11-05'),
        (5, 'Evan Wright', 'HR', 45000, '2021-09-30')
    ]
    cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?)', employees)
    conn.commit()
    conn.close()
    print("✅ Datenbank initialisiert.")

# --- TEIL 2: Der Agent ---
class SQLAgent:
    def __init__(self):
        print("⏳ Lade Modell (CPU)... das dauert ca. 1 Minute...")
        BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
        ADAPTER_ID = "DEIN_HF_NAME/Qwen2.5-SQL-Assistant-Prod" # <--- HIER DEINEN NAMEN!

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        # WICHTIG: Auf CPU nutzen wir float32 statt 4-bit, da stabiler
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            device_map="cpu", 
            torch_dtype=torch.float32
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

    def process_query(self, user_question):
        # 1. SQL Generieren
        schema = "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER, hire_date DATE)"
        messages = [
            {"role": "system", "content": "You are a SQL expert. Output only the SQL query."},
            {"role": "user", "content": f"{schema}\nQuestion: {user_question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt") # Kein .to("cuda") da CPU
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in full_text:
            sql_query = full_text.split("assistant")[-1].strip()
        else:
            sql_query = full_text

        # 2. SQL Ausführen
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()
            
            # Formatierung der Antwort
            return f"🧠 Gedanke (SQL):\n{sql_query}\n\n📊 Ergebnis aus Datenbank:\n{results}"
        except Exception as e:
            return f"❌ Fehler: {e}\n\nVersuchter SQL: {sql_query}"

# Initialisierung beim Start des Servers
setup_db()
agent = SQLAgent()

# --- TEIL 3: Die UI (Gradio Chat Interface) ---
def chat_response(message, history):
    return agent.process_query(message)

description = """
# 🤖 SQL Agent Demo
Dieser Agent übersetzt deine Fragen in SQL und **führt sie direkt auf einer Test-Datenbank aus**.
* Tabelle: `employees` (name, department, salary, hire_date)
* Probier es aus: "Who earns more than 80000?"
"""

demo = gr.ChatInterface(
    fn=chat_response,
    title="Autonomous SQL Agent",
    description=description,
    examples=["Show me all employees in Sales.", "Who earns the most?", "Count the employees in Engineering."],
    type="messages" 
)

demo.launch()