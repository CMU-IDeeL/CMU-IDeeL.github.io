import json
import os
import pathlib
import re
import subprocess
from datetime import datetime
from typing import Optional

import discord
from dotenv import load_dotenv
from ruamel.yaml import YAML

load_dotenv()

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
REPO_PATH = pathlib.Path(os.getenv("REPO_PATH", "").strip()).expanduser()
BRANCH = os.getenv("TARGET_BRANCH", "feature/f26-bot-automation").strip()
ALLOWED_CHANNEL_IDS = {int(x.strip()) for x in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if x.strip().isdigit()}
ALLOWED_UPLOADER_IDS = {int(x.strip()) for x in os.getenv("ALLOWED_UPLOADER_IDS", "").split(",") if x.strip().isdigit()}
SERVER_ID = int(os.getenv("SERVER_ID", "0") or "0")
PRIMARY_LECTURES_CHANNEL_ID = int(os.getenv("PRIMARY_LECTURES_CHANNEL_ID", "0") or "0")
HOLD_MINUTES = int(os.getenv("HOLD_MINUTES", "120"))
AUTO_COMMIT = os.getenv("AUTO_COMMIT", "true").lower() == "true"
AUTO_PUSH = os.getenv("AUTO_PUSH", "false").lower() == "true"

INBOX = pathlib.Path("F26/documents/slides/inbox")
MANIFEST = pathlib.Path("F26/automation/upload_manifest.jsonl")
LECTURES_YAML = pathlib.Path("F26/pages/tables_data/lectures.yaml")
ALLOWED_EXT = {".pdf", ".ppt", ".pptx"}
BAD_KEYWORDS = {"recitation", "lab", "hw", "homework", "bootcamp", "quiz", "exam", "solution", "draft"}
LECTURE_RE = re.compile(r"\blec(?:ture)?[\s._-]*0*(\d{1,2})\b", re.I)
COURSE_TOKEN_RE = re.compile(r"\b(f26|fall[\s._-]*26)\b", re.I)


def run_git(args):
    return subprocess.run(["git", *args], cwd=REPO_PATH, capture_output=True, text=True, check=False)


def validate():
    if not BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN missing")
    if not REPO_PATH.exists():
        raise RuntimeError("REPO_PATH missing/invalid")
    if not ALLOWED_CHANNEL_IDS:
        raise RuntimeError("ALLOWED_CHANNEL_IDS missing")
    if SERVER_ID <= 0:
        raise RuntimeError("SERVER_ID missing")
    if PRIMARY_LECTURES_CHANNEL_ID <= 0:
        raise RuntimeError("PRIMARY_LECTURES_CHANNEL_ID missing")


def expected_name_for_lecture(lecture_num: int) -> Optional[str]:
    yaml_abs = REPO_PATH / LECTURES_YAML
    if not yaml_abs.exists():
        return None
    yaml = YAML(typ="safe")
    data = yaml.load(yaml_abs.read_text(encoding="utf-8")) or {}
    for lec in data.get("lectures", []):
        if int(lec.get("number", -1)) != int(lecture_num):
            continue
        for entry in lec.get("slides_videos", []):
            if entry.get("text") != "Slides":
                continue
            url = str(entry.get("url", "")).strip()
            if url:
                return pathlib.Path(url).name
    return None


def classify(filename: str):
    lower = filename.lower()
    ext = pathlib.Path(lower).suffix
    if ext not in ALLOWED_EXT:
        return False, "unsupported_extension", None, None
    if any(k in lower for k in BAD_KEYWORDS):
        return False, "blocked_keyword", None, None
    m = LECTURE_RE.search(lower)
    if not m:
        return False, "missing_lecture_number", None, None
    lecture_num = int(m.group(1))
    canonical = expected_name_for_lecture(lecture_num)
    return True, "ok", lecture_num, canonical


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not message.guild or message.guild.id != SERVER_ID:
        return
    if not message.attachments:
        return

    inbox_abs = REPO_PATH / INBOX
    inbox_abs.mkdir(parents=True, exist_ok=True)
    manifest_abs = REPO_PATH / MANIFEST
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    for a in message.attachments:
        ok, reason, lecture_num, canonical_name = classify(a.filename)
        upload_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", a.filename)
        out_name = f"{upload_id}_{safe_name}"
        rel_path = INBOX / out_name
        abs_path = REPO_PATH / rel_path

        status = "rejected"
        final_reason = reason
        if ok:
            status = "pending_approval"
            if message.channel.id != PRIMARY_LECTURES_CHANNEL_ID:
                status = "needs_manual_review"
                final_reason = "non_primary_channel"
            if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
                status = "needs_manual_review"
                final_reason = "non_allowed_channel"
            if ALLOWED_UPLOADER_IDS and message.author.id not in ALLOWED_UPLOADER_IDS:
                status = "needs_manual_review"
                final_reason = "untrusted_uploader"
            if not COURSE_TOKEN_RE.search(a.filename):
                status = "needs_manual_review"
                final_reason = "missing_f26_token"

        if ok:
            await a.save(abs_path)
            saved += 1

        rec = {
            "upload_id": upload_id,
            "status": status,
            "reason": final_reason,
            "lecture_number": lecture_num,
            "original_name": a.filename,
            "saved_path": str(rel_path).replace("\\", "/") if ok else "",
            "canonical_name": canonical_name or "",
            "channel_id": message.channel.id,
            "message_id": message.id,
            "uploader_id": message.author.id,
            "uploaded_at_utc": datetime.utcnow().isoformat() + "Z",
            "hold_minutes": HOLD_MINUTES,
        }
        with manifest_abs.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if AUTO_COMMIT and saved:
        run_git(["add", str(INBOX).replace("\\", "/"), str(MANIFEST).replace("\\", "/")])
        run_git(["commit", "-m", f"F26 inbox: discord uploads from msg {message.id}"])
        if AUTO_PUSH:
            run_git(["push", "origin", BRANCH])

    await message.reply(
        f"Processed {len(message.attachments)} file(s). Saved {saved}. "
        f"Hold window: {HOLD_MINUTES} min. Non-lecture/non-prof uploads are flagged for manual review."
    )


if __name__ == "__main__":
    validate()
    client.run(BOT_TOKEN)
