# F26 Discord Ingestion Bot

This bot ingests uploads into `F26/documents/slides/inbox/` and logs metadata in `F26/automation/upload_manifest.jsonl`.

## Safety Rules

- Primary channel is `#lectures` (configured by `PRIMARY_LECTURES_CHANNEL_ID`).
- Uploads from other allowed channels are not dropped, but flagged `needs_manual_review`.
- Optional uploader allow-list (`ALLOWED_UPLOADER_IDS`) marks non-listed users as `needs_manual_review`.
- Rejects obvious out-of-context files (labs/HW/recitation/exam keywords, missing lecture number, unsupported extension).
- Holds files for a manual review window (`HOLD_MINUTES`).
- On approval, files are renamed to canonical lecture style from `F26/pages/tables_data/lectures.yaml` when available.

## Run

```powershell
cd C:\Users\shiro\OneDrive\Desktop\Robot\CMU-IDeeL.github.io\tools\f26_discord_bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python discord_ingest_bot.py
```

## Review Queue

```powershell
cd C:\Users\shiro\OneDrive\Desktop\Robot\CMU-IDeeL.github.io
python tools\f26_discord_bot\process_pending_uploads.py --list
python tools\f26_discord_bot\process_pending_uploads.py --approve <upload_id>
python tools\f26_discord_bot\process_pending_uploads.py --reject <upload_id> --reason wrong_slide
```

Approved files move to `F26/documents/slides/` and update `F26/pages/tables_data/lectures.yaml`.
