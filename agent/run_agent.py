import sqlite3
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class SQLAgent:
    def __init__(self, base_model_id, adapter_id, db_path):
        self.db_path = db_path
        print("🤖 Lade das Gehirn des Agenten...")
        
        # Modell laden
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_id)
        
    def generate_sql(self, question, schema_context):
        messages = [
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": f"{schema_context}\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrahiere alles nach 'assistant'
        if "assistant" in full_text:
            return full_text.split("assistant")[-1].strip()
        return full_text

    def execute_sql(self, query):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            return f"Fehler bei SQL-Ausführung: {e}"

    def run(self):
        schema = "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER, hire_date DATE)"
        print("\n✅ Agent bereit! Tippe 'exit' zum Beenden.")
        
        while True:
            # Hier wartet Colab auf deine Eingabe
            user_input = input("\nDeine Frage an die Datenbank: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Bis bald!")
                break
                
            # 1. Denken (SQL generieren)
            sql = self.generate_sql(user_input, schema)
            print(f"🧠 Gedanke (SQL): {sql}")
            
            # 2. Handeln (SQL ausführen)
            data = self.execute_sql(sql)
            
            # 3. Antworten
            print(f"📊 Ergebnis aus DB: {data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    args = parser.parse_args()

    agent = SQLAgent(
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_id=args.adapter,
        db_path="data/dummy_database.db"
    )
    agent.run()