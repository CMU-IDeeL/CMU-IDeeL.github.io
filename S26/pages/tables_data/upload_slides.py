import argparse
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ruamel.yaml import YAML

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
LECTURES_YAML = os.path.join(SCRIPT_DIR, "lectures.yaml")
SLIDES_DIR = os.path.join(ROOT_DIR, "documents", "slides")

STATUS_FOUND = "FOUND"
STATUS_MISSING = "MISSING"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_SKIPPED = "SKIPPED"


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=4, sequence=4, offset=2)
    return y


def parse_lecture_date(date_str: str, year: int = 2026) -> Optional[date]:
    """Parse either legacy format ('Monday <br> Feb 9') or ISO format ('2026-02-09')."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    if "<br>" in date_str:
        parts = date_str.split("<br>")
        if len(parts) < 2:
            return None
        month_day = parts[1].strip().replace(",", "").split()
        if len(month_day) < 2:
            return None
        month_str, day_token = month_day[0].strip(), month_day[1].strip()
        day = int(re.sub(r"[^0-9]", "", day_token))
        try:
            month = datetime.strptime(month_str, "%b").month
        except ValueError:
            month = datetime.strptime(month_str, "%B").month
        return date(year, month, day)
    return None


def generate_manifest(slides_dir: str) -> Dict[int, List[str]]:
    """Create deterministic mapping of lecture numbers to slide files."""
    manifest: Dict[int, List[str]] = {}
    pattern = re.compile(r"(?<![A-Za-z0-9])lec(?:ture)?[\s._-]*0*(\d+)(?!\d)", re.I)
    for name in sorted(os.listdir(slides_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(name)[0]
        m = pattern.search(base)
        if not m:
            continue
        num = int(m.group(1))
        manifest.setdefault(num, []).append(name)
    return manifest


def _rank_candidate(name: str, lecture_date: Optional[str]) -> int:
    """Lower score is better."""
    score = 100
    lowered = name.lower()
    if "v2" in lowered or "draft" in lowered:
        score += 20
    if lecture_date:
        compact = lecture_date.replace("-", "")
        if lecture_date in name or compact in name:
            score -= 30
    if lowered.startswith("lec") or lowered.startswith("lecture"):
        score -= 5
    return score


def find_matching_slide(
    lecture_number: int, lecture_date: str, slides_dir: str
) -> Tuple[str, Optional[str], List[str]]:
    """Return status, selected relative path or None, and candidate filenames."""
    manifest = generate_manifest(slides_dir)
    candidates = manifest.get(int(lecture_number), [])
    if not candidates:
        return STATUS_MISSING, None, []
    if len(candidates) == 1:
        return STATUS_FOUND, candidates[0], candidates
    ranked = sorted(candidates, key=lambda c: (_rank_candidate(c, lecture_date), c.lower()))
    best = ranked[0]
    # If tie at top rank, keep explicit ambiguity to avoid accidental wrong attachment.
    if len(ranked) > 1 and _rank_candidate(ranked[0], lecture_date) == _rank_candidate(
        ranked[1], lecture_date
    ):
        return STATUS_AMBIGUOUS, None, candidates
    return STATUS_FOUND, best, candidates


def _has_existing_slides_link(lecture: dict) -> bool:
    for item in lecture.get("slides_videos", []):
        if item.get("text") == "Slides" and item.get("url"):
            return True
    return False


def _insert_or_replace_slides_link(lecture: dict, url: str) -> None:
    slides_videos = lecture.get("slides_videos", [])
    for item in slides_videos:
        if item.get("text") == "Slides":
            item["url"] = url
            lecture["slides_videos"] = slides_videos
            return
    slides_videos.insert(0, {"text": "Slides", "url": url})
    lecture["slides_videos"] = slides_videos


def update_yaml_links(
    lectures_yaml: str,
    manifest: dict,
    dry_run: bool = False,
    update_all: bool = False,
    year: int = 2026,
) -> dict:
    """Update slide links in YAML; return status counts."""
    y = _yaml()
    with open(lectures_yaml, "r", encoding="utf-8") as f:
        data = y.load(f)
    lectures = data.get("lectures", [])

    timezone = ZoneInfo("America/New_York")
    today = datetime.now(timezone).date()

    counts = {STATUS_FOUND: 0, STATUS_MISSING: 0, STATUS_AMBIGUOUS: 0, STATUS_SKIPPED: 0}
    changed = False

    for lecture in lectures:
        lec_num = lecture.get("number")
        if lec_num in (None, "", 0):
            counts[STATUS_SKIPPED] += 1
            continue
        lec_date_obj = parse_lecture_date(lecture.get("date", ""), year=year)
        if lec_date_obj is None:
            print(f"[WARN] Could not parse date for lecture {lec_num}; skipping")
            counts[STATUS_SKIPPED] += 1
            continue
        if not update_all and lec_date_obj != today:
            continue
        if _has_existing_slides_link(lecture):
            print(f"[INFO] Lecture {lec_num}: slide link already present, skipping")
            counts[STATUS_SKIPPED] += 1
            continue

        status, selected, candidates = find_matching_slide(
            int(lec_num), lec_date_obj.isoformat(), SLIDES_DIR
        )
        if status == STATUS_FOUND and selected:
            rel_path = os.path.relpath(os.path.join(SLIDES_DIR, selected), ROOT_DIR).replace("\\", "/")
            print(f"[FOUND] Lecture {lec_num}: {rel_path}")
            _insert_or_replace_slides_link(lecture, rel_path)
            counts[STATUS_FOUND] += 1
            changed = True
        elif status == STATUS_AMBIGUOUS:
            print(f"[AMBIGUOUS] Lecture {lec_num}: candidates={candidates}")
            counts[STATUS_AMBIGUOUS] += 1
        else:
            print(f"[MISSING] Lecture {lec_num}: no matching slide found")
            counts[STATUS_MISSING] += 1

    if changed and not dry_run:
        with open(lectures_yaml, "w", encoding="utf-8") as f:
            y.dump(data, f)
        print("[INFO] lectures.yaml updated")
    elif changed and dry_run:
        print("[INFO] Dry-run enabled: no files were modified")

    return counts


def upload(dry_run: bool = False, update_all: bool = False, year: int = 2026) -> int:
    if not os.path.isfile(LECTURES_YAML):
        print(f"[ERROR] lectures.yaml not found at: {LECTURES_YAML}")
        return 1
    if not os.path.isdir(SLIDES_DIR):
        print(f"[ERROR] slides directory not found at: {SLIDES_DIR}")
        return 1

    manifest = generate_manifest(SLIDES_DIR)
    counts = update_yaml_links(
        lectures_yaml=LECTURES_YAML,
        manifest=manifest,
        dry_run=dry_run,
        update_all=update_all,
        year=year,
    )

    print("========================================")
    for key in (STATUS_FOUND, STATUS_MISSING, STATUS_AMBIGUOUS, STATUS_SKIPPED):
        print(f"{key}: {counts[key]}")
    print("========================================")

    # Fail closed on ambiguity in automated runs.
    if counts[STATUS_AMBIGUOUS] > 0:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload lecture slides into lectures.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to lectures.yaml")
    parser.add_argument(
        "--update-all",
        action="store_true",
        help="Process all lectures missing slide links (default: only today's lecture)",
    )
    parser.add_argument("--year", type=int, default=2026, help="Default year for legacy date parsing")
    args = parser.parse_args()
    return upload(dry_run=args.dry_run, update_all=args.update_all, year=args.year)


if __name__ == "__main__":
    raise SystemExit(main())
