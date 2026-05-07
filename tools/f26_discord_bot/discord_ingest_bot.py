import json
import os
import pathlib
import re
import subprocess
from datetime import date, datetime
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


def parse_yaml_date(date_str: str, year: int = 2026) -> Optional[date]:
    if "<br>" not in date_str:
        return None
    part = date_str.split("<br>", 1)[1].strip()
    bits = part.split()
    if len(bits) < 2:
        return None
    month_str, day_str = bits[0], bits[1]
    try:
        day_num = int(day_str)
    except ValueError:
        return None
    try:
        month_num = datetime.strptime(month_str, "%b").month
    except ValueError:
        try:
            month_num = datetime.strptime(month_str, "%B").month
        except ValueError:
            return None
    return date(year, month_num, day_num)


def infer_lecture_from_schedule() -> tuple[Optional[int], Optional[str], str]:
    yaml_abs = REPO_PATH / LECTURES_YAML
    if not yaml_abs.exists():
        return None, None, "no_yaml"
    yaml = YAML(typ="safe")
    data = yaml.load(yaml_abs.read_text(encoding="utf-8")) or {}
    lectures = data.get("lectures", [])
    today = datetime.now().date()

    def has_slides(lec: dict) -> bool:
        for entry in lec.get("slides_videos", []):
            if entry.get("text") == "Slides" and str(entry.get("url", "")).strip():
                return True
        return False

    # Prefer lectures dated <= today that have no slide link yet.
    candidates = []
    for lec in lectures:
        num = lec.get("number")
        if num is None:
            continue
        d = parse_yaml_date(str(lec.get("date", "")))
        if d is None:
            continue
        if d <= today and not has_slides(lec):
            candidates.append((d, int(num)))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        num = candidates[0][1]
        return num, expected_name_for_lecture(num), "inferred_latest_unfilled"

    # If all lectures already have slides (template copied), fallback to latest past lecture.
    fallback = []
    for lec in lectures:
        num = lec.get("number")
        if num is None:
            continue
        d = parse_yaml_date(str(lec.get("date", "")))
        if d is None:
            continue
        if d <= today:
            fallback.append((d, int(num)))
    if fallback:
        fallback.sort(key=lambda x: x[0], reverse=True)
        num = fallback[0][1]
        return num, expected_name_for_lecture(num), "inferred_latest_dated"

    return None, None, "no_candidate"


def classify(filename: str):
    lower = filename.lower()
    ext = pathlib.Path(lower).suffix
    if ext not in ALLOWED_EXT:
        return "reject", "unsupported_extension", None, None
    if any(k in lower for k in BAD_KEYWORDS):
        return "reject", "blocked_keyword", None, None
    m = LECTURE_RE.search(lower)
    if not m:
        return "manual", "missing_lecture_number", None, None
    lecture_num = int(m.group(1))
    canonical = expected_name_for_lecture(lecture_num)
    return "ok", "ok", lecture_num, canonical


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
        kind, reason, lecture_num, canonical_name = classify(a.filename)
        upload_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", a.filename)
        out_name = f"{upload_id}_{safe_name}"
        rel_path = INBOX / out_name
        abs_path = REPO_PATH / rel_path

        status = "rejected"
        final_reason = reason
        should_save = False
        if kind in {"ok", "manual"}:
            should_save = True
            status = "pending_approval" if kind == "ok" else "needs_manual_review"
            final_reason = "ok" if kind == "ok" else reason
        if kind == "manual" and reason == "missing_lecture_number":
            inferred_num, inferred_name, inferred_reason = infer_lecture_from_schedule()
            if inferred_num is not None:
                lecture_num = inferred_num
                canonical_name = inferred_name
                status = "pending_approval"
                final_reason = inferred_reason
            else:
                status = "needs_manual_review"
                final_reason = inferred_reason
        if kind == "ok":
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

        if should_save:
            await a.save(abs_path)
            saved += 1

        rec = {
            "upload_id": upload_id,
            "status": status,
            "reason": final_reason,
            "lecture_number": lecture_num,
            "original_name": a.filename,
            "saved_path": str(rel_path).replace("\\", "/") if should_save else "",
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
