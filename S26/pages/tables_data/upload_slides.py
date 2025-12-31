import os
import re
import subprocess
from ruamel.yaml import YAML
from datetime import datetime, date, timedelta, time
from zoneinfo import ZoneInfo

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir)) # S26/
LECTURES_YAML = os.path.join(SCRIPT_DIR, "lectures.yaml") # .../S26/pages/lectures.yaml
SLIDES_DIR    = os.path.join(ROOT_DIR, "documents", "slides") # .../S26/documents/slides/

def parse_lecture_date(date_str, year = 2026):
    # this function parses the lecture date and returns date and month of the lecture
    # this function assumes that the date argument has string in format "[weekday] <br> [month] [day]". Example: "Monday <br> Feb 9"
    # [month] can be in full form or abbreviated form: Feb or February
    # [date] has to be a number (specifically integer)
    parts = date_str.split("<br>")
    month_day = parts[1].strip().split()
    month_str, day = month_day[0].strip(), int(month_day[1].strip())
    try:
        month = datetime.strptime(month_str, "%b").month
    except ValueError:
        month = datetime.strptime(month_str, "%B").month
    return date(year, month, day)


def find_slide_by_lecture_number(lecture_num):
    # this function gets the path where the lecture slide is stored using the lecture number
    # this function assumes that the lecture number is part of file name (parsed through regex )
    n = int(lecture_num)
    pattern = re.compile(rf"(?<![A-Za-z0-9])lec(?:ture)?[\s._-]*0*{n}(?!\d)", re.I)
    for name in os.listdir(SLIDES_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(name)[0]
        if pattern.search(base):
            print(f"[INFO] Found lecture slides (based on lecture number): {name}")
            return name  
    return None


def find_slide_by_commit_time(start_timestamp):
    best_match = None
    recent = -1
    for name in os.listdir(SLIDES_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(SLIDES_DIR, name)
        out = subprocess.check_output(["git", "log", "-1", "--format=%ct", "--", full_path], cwd=ROOT_DIR, text=True).strip()  
        file_commit_time = int(out) 
        if file_commit_time >= start_timestamp and file_commit_time > recent:
            recent = file_commit_time
            best_match = name
    print(f"[INFO] Found lecture slides (based on commit time): {best_match}")
    return best_match


def upload():
    # this function assumes that it will be invoked only on Monday's and Wednesday's
    # this function assume each lecture enty in lectures.yaml has slides_videos entry (either empty [] or filled with something)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=2)

    if not os.path.isfile(LECTURES_YAML):
        print(f"[ERROR] lectures.yaml not found at the path: {LECTURES_YAML}")
        return
    if not os.path.isdir(SLIDES_DIR):
        print(f"[ERROR] Slides directory not found at the path: {SLIDES_DIR}")
        return
    
    timezone = ZoneInfo("America/New_York")
    now = datetime.now(timezone)
    today = now.date()
    today = date(2026, 1, 12) # for testing

    with open(LECTURES_YAML, "r") as f:
        data = yaml.load(f)
    lectures = data.get("lectures", [])

    target = None
    lec_num = None
    for lec in lectures:
        if lec.get("number", "") == 0: # skipping lecture 0 as it doesnt have tedious slide upload task
            continue
        lec_date = parse_lecture_date(lec.get("date", ""))
        if lec_date == today:
            target = lec
            lec_num = lec.get("number")
            break
    
    if target is None or lec_num is None:
        print("[ERROR] Could not find today's lecture in YAML :(")
        return
    print(f"[INFO] Lecture Number: {lec_num}")
    print(f"[INFO] Parsed lecture date: {lec_date}")
    
    slide_placeholder = target.get("slides_videos")
    for item in slide_placeholder:
        if item.get("text") == "Slides" and item.get("url"):
            print("[WARN] Slides already linked in the website")
            return

    pdf_name = find_slide_by_lecture_number(lec_num)
    if not pdf_name: # Fallback: if the lecture slide is not found based on lecture number, we take the most recently committed file to GitHub in slides folder
        print(f"[WARN] Not able to find lecture slide based on lecture number; Trying to find based on commit time")
        if today.weekday() == 0: # for Monday lecture, check from Thursday 
            start_day = today - timedelta(days=4)
        elif today.weekday() == 2: # for Wednesday lecture, check from Tuesday 
            start_day = today - timedelta(days=1)
        start_timestamp = datetime.combine(start_day, time(0, 0), timezone).timestamp()
        pdf_name = find_slide_by_commit_time(start_timestamp)
    if not pdf_name:
        print(f"[WARN] Not able to find lecture slide based on commit time")
        print(f"[ERROR] No slides found for the lecture dated {today} with lecture number: {lec_num}")
        return
    pdf_path = os.path.join(SLIDES_DIR, pdf_name)
    print(f"[INFO] Absolute lecture slide path: {pdf_path}")
    rel_pdf_path = os.path.relpath(pdf_path, start=ROOT_DIR).replace("\\", "/")
    print(f"[INFO] Relative lecture slide path: {rel_pdf_path}")
    target["slides_videos"].insert(0, {"text": "Slides", "url": rel_pdf_path})

    with open(LECTURES_YAML, "w") as f:
        yaml.dump(data, f)
    print(f"[INFO] Added slides path for the lecture dated {today} with lecture number: {lec_num} as {rel_pdf_path}")

if __name__ == "__main__":
    upload()