# 🤖 SQL Assistant Agent (Qwen 2.5 Fine-Tune)

Ein Technical Assistant Agent, der natürliche Sprache in SQL übersetzt und auf einer Datenbank ausführt. Trainiert mit QLoRA auf Qwen-2.5-1.5B.

[🔗 **Live Demo auf Hugging Face Spaces**](HIER_DEIN_LINK_ZUM_SPACE_EINFÜGEN)
[🤗 **Model Card**](HIER_DEIN_LINK_ZUM_MODELL_EINFÜGEN)

## 🎯 Features
- **Text-to-SQL:** Übersetzt Fragen ("Who works in Sales?") in SQL.
- **Agentic Workflow:** Führt Queries autonom auf einer SQLite-Datenbank aus.
- **Efficient Fine-Tuning:** Trainiert auf einer T4 GPU (Colab) mittels PEFT/LoRA.

## 🛠️ Installation & Setup

1. **Repo klonen**
   ```bash
   git clone https://github.com/DEIN_USER/sql-assistant.git
   cd sql-assistant
   pip install -r requirements.txt