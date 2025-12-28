import argparse
from huggingface_hub import HfApi, create_repo

def main(args):
    api = HfApi()
    
    # 1. Repo-Namen bauen
    full_repo_id = f"{args.username}/{args.repo_name}"
    print(f"Ziel-Repository: {full_repo_id}")

    # 2. Repository erstellen (falls es noch nicht existiert)
    try:
        create_repo(full_repo_id, repo_type="model", exist_ok=True)
        print("Repository gefunden oder erstellt.")
    except Exception as e:
        print(f"Hinweis beim Repo-Erstellen: {e}")

    # 3. Dateien hochladen
    print(f"Lade Ordner '{args.model_dir}' hoch... Bitte warten.")
    
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=full_repo_id,
        repo_type="model",
        commit_message=f"Upload model from production script: {args.repo_name}"
    )
    
    print("\n✅ Upload erfolgreich!")
    print(f"Dein Modell ist hier: https://huggingface.co/{full_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True, help="Hugging Face Nutzername")
    parser.add_argument("--repo_name", type=str, required=True, help="Name für das neue Modell auf HF")
    parser.add_argument("--model_dir", type=str, default="./outputs/final_model", help="Lokaler Pfad zum Modell")
    
    args = parser.parse_args()
    main(args)