import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize_sql(query):
    """Bereinigt SQL von Leerzeichen und Groß/Kleinschreibung für fairen Vergleich"""
    if not query: return ""
    query = query.lower().replace(";", "").replace("\n", " ")
    return " ".join(query.split())

def main(args):
    # 1. Basis-Modell & Adapter laden
    print(f"Lade Basis-Modell: {args.base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, device_map="auto", torch_dtype=torch.float16
    )
    
    print(f"Lade Adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # 2. Test-Daten laden (Die letzten 100 Zeilen des Datasets als Testset nehmen)
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    test_dataset = dataset.select(range(len(dataset)-args.num_samples, len(dataset)))

    correct_count = 0
    total_count = 0

    print(f"Starte Evaluation auf {args.num_samples} Beispielen...")

    for sample in tqdm(test_dataset):
        # Prompt bauen
        messages = [
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": f"{sample['context']}\nQuestion: {sample['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generieren
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        # Antwort extrahieren
        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sql = generated_full.split("assistant")[-1].strip()
        
        # Vergleichen (Normalized Exact Match)
        truth_norm = normalize_sql(sample["answer"])
        pred_norm = normalize_sql(generated_sql)

        if truth_norm == pred_norm:
            correct_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100
    print(f"\n==========================================")
    print(f"RESULTAT: Exact Match Accuracy: {accuracy:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    main(args)



