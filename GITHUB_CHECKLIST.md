# GitHub Release Checklist

Use this checklist before publishing the project.

## 1. Verify local checks

```powershell
python -m py_compile main_telegram_database.py dashboard_streamlit.py record.py fall_features.py tests/smoke_test.py
python scripts/migrate_database.py
python tests/smoke_test.py
```

Run the dashboard locally:

```powershell
streamlit run dashboard_streamlit.py
```

Check these tabs:

- `Overview`: latest event and timeline render correctly.
- `Events`: table shows status, review time and review note.
- `Manage`: single event review and batch review work.
- `Reports`: daily, status and camera summaries render correctly.
- `Models`: sequence comparison metrics render correctly.

## 2. Do not publish secrets or large local files

Do not commit:

- `telegram_config.json`
- `.env`
- `fall_events.db`
- `captures/`
- `recordings/`
- `datasets/`
- `datasets Lei2Fall/`
- `datasets MCFD/`
- `datasets URFD/`
- `ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-*/`
- `frame_feature_cache*/`
- `.venv/`
- `__pycache__/`
- `.codex_review/`
- `.codex/`
- `.agents/`
- `anaconda_projects/`
- `*.ipynb`
- `*.pptx`
- Root-level `seq40/`
- `*.h5`
- `*.keras`
- `*.pt`

If the Telegram token was ever shared or pushed, regenerate the bot token before publishing.

## 3. Files that should be included

Recommended public files:

- `README.md`
- `GITHUB_CHECKLIST.md`
- `.gitignore`
- `.env.example`
- `requirements.txt`
- `telegram_config.example.json`
- `models_lstm/lstm_runtime_config.example.json`
- `main_telegram_database.py`
- `dashboard_streamlit.py`
- `record.py`
- `fall_features.py`
- `extract_frame_feature_cache.py`
- `build_sequence_dataset_from_frame_cache.py`
- `compare_lstm_transformer.py`
- `tests/smoke_test.py`
- `scripts/migrate_database.py`
- Small result files such as `model_comparison.json` and `models_4_clean/sequence_comparison_30_35_40.csv`
- Small images of training curves/confusion matrices, if useful for the README

## 4. Model artifact note

The repository ignores `.h5`, `.keras` and `.pt` by default. For a public GitHub repo, choose one:

- Use GitHub Releases or Google Drive for model files and put the download link in `README.md`.
- Use Git LFS for model files.
- Keep the repo code-only and document where users should place model files locally.

Expected local model path:

```text
models_4_clean/seq30/lstm_best_seq30.h5
```

Expected YOLO pose model path:

```text
yolo11m-pose.pt
```

## 5. Git commands

Check repository state:

```powershell
git status
```

If this folder is not a valid Git repository yet, initialize it:

```powershell
git init
git branch -M main
```

Preview what will be committed:

```powershell
git status --short
```

Add safe files explicitly:

```powershell
git add README.md GITHUB_CHECKLIST.md .gitignore .env.example requirements.txt telegram_config.example.json
git add main_telegram_database.py dashboard_streamlit.py record.py fall_features.py
git add extract_frame_feature_cache.py build_sequence_dataset_from_frame_cache.py compare_lstm_transformer.py
git add tests/smoke_test.py models_lstm/lstm_runtime_config.example.json
git add model_comparison.json models_4_clean/sequence_comparison_30_35_40.csv
```

Commit:

```powershell
git commit -m "Prepare fall detection monitoring system for GitHub"
```

Connect GitHub remote:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
