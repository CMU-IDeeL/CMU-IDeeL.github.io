import argparse
import json
import pathlib
import re
from datetime import datetime, timedelta, timezone

from ruamel.yaml import YAML

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO / "F26/automation/upload_manifest.jsonl"
SLIDES_DIR = REPO / "F26/documents/slides"
LECTURES_YAML = REPO / "F26/pages/tables_data/lectures.yaml"


def load_records():
    if not MANIFEST.exists():
        return []
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_records(rows):
    MANIFEST.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def eligible(rec):
    if rec.get("status") not in {"pending_approval", "needs_manual_review"}:
        return False
    t = datetime.fromisoformat(rec["uploaded_at_utc"].replace("Z", "+00:00"))
    hold = int(rec.get("hold_minutes", 120))
    return datetime.now(timezone.utc) >= t + timedelta(minutes=hold)


def update_yaml(lecture_number, filename):
    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(LECTURES_YAML.read_text(encoding="utf-8"))
    lectures = data.get("lectures", [])
    target = None
    for lec in lectures:
        if int(lec.get("number", -1)) == int(lecture_number):
            target = lec
            break
    if target is None:
        return False, "lecture_not_found"

    rel = f"./documents/slides/{filename}"
    slots = target.get("slides_videos", [])
    for s in slots:
        if s.get("text") == "Slides":
            s["url"] = rel
            LECTURES_YAML.write_text("", encoding="utf-8")
            with LECTURES_YAML.open("w", encoding="utf-8") as f:
                yaml.dump(data, f)
            return True, "updated_existing"

    slots.insert(0, {"text": "Slides", "url": rel})
    target["slides_videos"] = slots
    LECTURES_YAML.write_text("", encoding="utf-8")
    with LECTURES_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return True, "inserted"


def approve(upload_id):
    rows = load_records()
    found = None
    for r in rows:
        if r.get("upload_id") == upload_id:
            found = r
            break
    if not found:
        print("upload_id not found")
        return
    if not eligible(found):
        print("hold window not complete or status not pending")
        return

    src = REPO / found["saved_path"]
    if not src.exists():
        found["status"] = "error"
        found["reason"] = "missing_source"
        write_records(rows)
        print("source file missing")
        return

    clean = found.get("canonical_name", "").strip() or found.get("original_name", "").strip() or re.sub(r"^[A-Za-z0-9]+_", "", src.name)
    dst = SLIDES_DIR / clean
    src.replace(dst)

    ok, reason = update_yaml(found["lecture_number"], clean)
    found["status"] = "approved" if ok else "error"
    found["reason"] = reason
    found["approved_at_utc"] = datetime.now(timezone.utc).isoformat()
    found["final_path"] = f"F26/documents/slides/{clean}"
    write_records(rows)
    print(f"approved: {upload_id} -> {dst.name} ({reason})")


def auto_approve_eligible():
    rows = load_records()
    approved = 0
    skipped = 0
    for r in rows:
        if r.get("status") != "pending_approval":
            skipped += 1
            continue
        if not eligible(r):
            skipped += 1
            continue
        upload_id = r.get("upload_id")
        if not upload_id:
            skipped += 1
            continue
        approve(upload_id)
        approved += 1
    print(f"auto_approve_summary approved={approved} skipped={skipped}")


def reject(upload_id, reason):
    rows = load_records()
    found = None
    for r in rows:
        if r.get("upload_id") == upload_id:
            found = r
            break
    if not found:
        print("upload_id not found")
        return
    if found.get("status") == "approved":
        print("already approved, cannot reject")
        return
    found["status"] = "rejected"
    found["reason"] = reason
    found["rejected_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_records(rows)
    print(f"rejected: {upload_id} ({reason})")


def list_pending():
    rows = load_records()
    for r in rows:
        if r.get("status") in {"pending_approval", "needs_manual_review"}:
            print(
                f"{r['upload_id']} status={r.get('status')} lecture={r.get('lecture_number')} "
                f"file={r.get('original_name')} eligible={eligible(r)} reason={r.get('reason')}"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--approve")
    p.add_argument("--auto-approve-eligible", action="store_true")
    p.add_argument("--reject")
    p.add_argument("--reason", default="manual_reject")
    args = p.parse_args()

    if args.list:
        list_pending()
    elif args.approve:
        approve(args.approve)
    elif args.auto_approve_eligible:
        auto_approve_eligible()
    elif args.reject:
        reject(args.reject, args.reason)
    else:
        p.print_help()
