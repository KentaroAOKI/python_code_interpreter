# Repository Guidelines

## Project Structure & Module Organization
Core orchestration lives in `python_code_interpreter.py`, which coordinates OpenAI/Azure responses and streams code into `python_code_notebook.py` (Papermill + Jupyter helper). Support scripts include `set_matplotlib_japanese_font.py` for locale-safe plots and `papermill_enhancement/` for future kernels tweaks. Reference data is under `sample_data/` (for example, `diagnosis.csv`), while generated notebooks and JSON transcripts land in `sample_results/` or the runtime `results/` folder. Dependency locks are in `requirements.txt`, user-specific credentials (including `PYCODEI_CLIENT` for OpenAI vs Azure selection) live in `~/.pycodei/config.json`, and optional guardrails live in `PYCODEI.md` (looked up in the config dir, install dir, or current workspace).

## Build, Test, and Development Commands
Install or refresh dependencies and user config before running:
```bash
pip install -e .        # exposes the pycodei console script
pycodei --help          # creates ~/.pycodei/config.json with placeholders
```
Launch the interpreter loop (prompts you before each code execution):
```bash
pycodei "Analyze diagnosis.csv with the breast cancer model"
```
To inspect or re-run a specific notebook in VS Code or Papermill, point directly to the artifact in `sample_results/*.ipynb`.

## Coding Style & Naming Conventions
Follow standard PEP 8 conventions (4-space indentation, snake_case for functions/variables, CapWords for classes). Keep configuration constants (paths, prefixes) near the top of the module and guard new env lookups with `os.getenv`. When adding notebooks or JSON outputs, use the existing `result_YYYYMMDD-HHMMSS` prefix to keep automation compatible. Prefer explicit imports and avoid wildcard imports to keep the tool-calling context predictable.

## Testing Guidelines
There is no formal test harness yet—verification relies on running realistic prompts through `python_code_interpreter.py`, reviewing the console trace, and opening the emitted notebook to confirm plots, tables, and error cells. When adding capabilities, craft a regression scenario in `sample_data/` and document the intended prompt/result pair in `sample_results/` to keep reviewers aligned. Keep notebooks lean (remove exploratory cells) so diffs stay reviewable.

## Commit & Pull Request Guidelines
Recent commits are short, imperative summaries (e.g., “Add sample results”, “Support for o3-mini”), so follow that style and group related edits together. Every PR should include: goal-oriented description, mention of any config key changes (`~/.pycodei/config.json`), reproduction steps (`pycodei "<prompt>"`), and screenshots or notebook snippets if UI/output changes. Link issues where applicable and call out breaking changes or manual migration steps (such as re-linking `/mnt/data`). Reviewers expect clean `git status` with only relevant artifacts tracked—do not commit personal notebooks or secrets.
