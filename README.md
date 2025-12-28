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

The project is organized into modular components for training, inference, and deployment:

```text
├── agent/
│   └── run_agent.py          # CLI Agent: Translates questions & executes SQL on local DB
│
├── data/
│   └── README.md             # Documentation for the dummy database generation
│
├── demo/
│   ├── app.py                # Gradio Web UI code (deployed on Hugging Face Spaces)
│   └── requirements.txt      # Lightweight dependencies for the CPU-only demo
│
├── notebooks/
│   ├── SQL_Assistant_Production.ipynb  # The main "Command Center" for Colab execution
│   └── sql_assistant.ipynb             # Initial exploration and experimental code
│
├── scripts/
│   ├── train.py              # PEFT/QLoRA Fine-Tuning script (w/ W&B logging)
│   ├── evaluate.py           # Evaluation script (Exact Match Metric)
│   ├── setup_db.py           # Generator script for the dummy SQLite database
│   └── deploy.py             # Automation script to upload models to Hugging Face
│
├── .gitignore
├── README.md
└── requirements.txt          # Core dependencies for training/reproduction
