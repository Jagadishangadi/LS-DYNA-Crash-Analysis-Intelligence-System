# LS-DYNA Crash Analysis Intelligence System

This repository contains a prototype Python-based analysis system for LS-DYNA crash simulations with AI-assisted reporting.

## Contents

- `ls_dyna_ai_analyzer.py` — main example analyzer and runner

## Requirements

Create a Python 3.10+ virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Environment variables:

- `ANTHROPIC_API_KEY` — required if using the Anthropic client

## Quick start

Run the example analysis (demo):

```bash
python ls_dyna_ai_analyzer.py
```

## Notes

- This project is a demo; the Anthropic integration requires proper API key setup and may need SDK/API adjustments depending on the installed `anthropic` package version.
- Feel free to open an issue or PR with improvements.
