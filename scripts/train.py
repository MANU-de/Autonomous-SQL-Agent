import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main(args):
    # Wandb Setup - Automatically use existing account (option 2)
    # Set environment to use existing credentials
    os.environ["WANDB_MODE"] = "online"
    # If wandb is not already logged in, this will use existing credentials from ~/.netrc or environment
    try:
        import wandb
        # Check if wandb is already initialized
        if not wandb.run:
            wandb.init(project="sql-assistant", mode="online", reinit=True)
    except Exception as e:
        print(f"Wandb initialization note: {e}")
        # Continue without wandb if there's an issue
    # 1. Datensatz laden
    print(f"Lade Datensatz: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    # Nur für Demo-Zwecke verkürzen, falls gewünscht
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    
    # 2. Modell & Tokenizer laden (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Lade Modell: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA Config - für Qwen2.5 Modelle
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 4. Model für k-bit training vorbereiten und PEFT anwenden
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # 5. Training Arguments - Kombiniere TrainingArguments mit SFTConfig
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=10,
        num_train_epochs=args.epochs,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=512,
        packing=False
    )

    # 6. Daten formatieren (Qwen Template)
    def format_prompt(sample):
        prompt = f"<|im_start|>system\nYou are a SQL expert.<|im_end|>\n<|im_start|>user\n{sample['context']}\nQuestion: {sample['question']}<|im_end|>\n<|im_start|>assistant\n{sample['answer']}<|im_end|>"
        return {"text": prompt}

    train_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    # 7. Trainer Starten (ohne peft_config, da Modell bereits PEFT-wrapped ist)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args
    )

    print("Starte Training...")
    trainer.train()
    
    print(f"Speichere Adapter nach {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--output_dir", type=str, default="./outputs/final_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=500) # Klein halten für Test
    
    args = parser.parse_args()
    main(args)

