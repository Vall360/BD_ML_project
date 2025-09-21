# Repository Guidelines

## Project Structure & Module Organization
Keep the main Python runners in the repository root. Use `bd_ml_project_light.py` for the streamlined pipeline, `bd_ml_project_local.py` when you need the full modelling stack, and leave `bd&ml_project.py` untouched as the original Colab export. Source data (`News_dataset.csv`, `BERT_project_OHE.csv`, `BERT_project_results.csv`, `Sources_class.xlsx`) and generated artefacts (`df_stock.csv`, `Report0.html`) live alongside the scripts; clean up scratch exports and `__pycache__/` before committing. If you rely on the optional relevance model, place it at `./Relevance_model/`.

Do not interact or read `bd_ml_project_local.py`, `bd&ml_project.py` unless specifically requested.


## Build, Test, and Development Commands
Create an isolated environment before running any script: `python3 -m venv .venv && source .venv/bin/activate`. Install the lightweight dependencies with `pip install pandas numpy scipy yfinance linearmodels stargazer`. 

When working on the full pipeline, add the NLP stack: `pip install torch transformers optuna lightgbm spacy datasets evaluate tqdm`. Run the end-to-end lightweight flow via `python3 bd_ml_project_light.py`; switch to `python3 bd_ml_project_local.py` if you need to regenerate the One-Hot encoding or touch modelling logic.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and descriptive `snake_case` function and variable names. Keep module-level constants uppercase (`BASE_DIR`, `USE_PRECOMPUTED_OHE`) and route paths through `pathlib.Path`. Prefer explicit helper functions over inline notebooks-style code; add short docstrings or comments only when behaviour is non-obvious.

## Testing Guidelines
There is no automated test suite yet, so validation is runtime-driven. After any change, run the relevant script and confirm it finishes without traceback, then spot-check regenerated artefacts (`df_stock.csv`, `Report0.html`) for anomalies. For data-transform edits, compare summaries (e.g., `df.describe()` in a scratch shell or notebook) against the previous version and record notable shifts in your PR description.

## Commit & Pull Request Guidelines
Match the existing history by using short, imperative commit subjects ("Add local runner", "Fix relevance fallback") and keep each commit focused on a single concern. In PRs, summarise the change, list the commands you executed (including dataset refreshes), and call out any required external files or toggles (such as `USE_PRECOMPUTED_OHE`). Attach metric tables or screenshots only when they clarify modelling changes.

## Data & Configuration Notes
Never commit proprietary or oversized CSV/XLSX files unless they are canonical inputs; add new filenames to `.gitignore` first. Document toggles you set in `bd_ml_project_local.py`, especially whether precomputed OHE inputs were reused. If you bundle a `Relevance_model/` directory for relevance scoring, verify its licences and keep the footprint minimal.
