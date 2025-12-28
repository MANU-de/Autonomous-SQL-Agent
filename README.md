# 🤖 SQL Assistant Agent (Qwen 2.5 Fine-Tune)

> An autonomous technical assistant that translates natural language questions into executable SQL queries.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/manuelaschrittwieser/SQL-Assistant-Prod)
[![Model](https://img.shields.io/badge/🤗%20Model-Qwen2.5--SQL--Assistant-blue)](https://huggingface.co/manuelaschrittwieser/Qwen2.5-SQL-Assistant-Prod)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Project Overview

This project implements an end-to-end **Text-to-SQL system**. It fine-tunes a **Qwen 2.5 (1.5B)** Large Language Model using **PEFT (LoRA)** to specialize in generating syntactically correct SQL queries from natural language prompts.

Beyond just generation, this repository features a fully functional **autonomous agent** capable of executing generated queries on a live SQLite database and a **production-ready web UI**.

### Key Features
*   **Precision Fine-Tuning:** Adapted on the `b-mc2/sql-create-context` dataset using QLoRA 4-bit quantization.
*   **Autonomous Agent:** A Python agent that thinks (generates SQL), acts (executes on DB), and responds with data.
*   **Live Monitoring:** Integrated with **Weights & Biases (W&B)** for tracking training metrics.
*   **Reproducible Pipeline:** Modular scripts for data prep, training, evaluation, and deployment.

---

## 📂 Repository Structure

```text
sql-assistant/
│
├── scripts/                 # Production pipeline scripts
│   ├── train.py             # QLoRA fine-tuning logic (w/ W&B logging)
│   ├── evaluate.py          # Normalized Exact Match evaluation
│   ├── setup_db.py          # Generates the dummy SQLite database
│   └── deploy.py            # Automation script for HF Hub uploads
│
├── agent/                   # Autonomous Agent Logic
│   └── run_agent.py         # CLI Agent that executes SQL on local DB
│
├── deployment/              # Deployment Configuration
│   ├── app.py               # Gradio Web UI Code (for Hugging Face Spaces)
│   └── requirements.txt     # Dependencies for the deployment environment
│
├── notebooks/               # Experimental & Exploration
│   ├── sql_assistant.ipynb    # Initial research and data analysis
│   └── SQL_Assistant_Production.ipynb   # Launcher notebook for Google Colab
│
└── requirements.txt         # Core dependencies for reproduction
