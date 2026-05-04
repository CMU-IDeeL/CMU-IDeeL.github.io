import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_link_item(item: Any) -> bool:
    return isinstance(item, dict) and _is_non_empty_string(item.get("text")) and _is_non_empty_string(
        item.get("url")
    )


def check_date_format(date_str: str) -> bool:
    """Ensure date is YYYY-MM-DD and logically valid, or legacy semester format."""
    if not _is_non_empty_string(date_str):
        return False
    date_str = date_str.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    # Legacy support, e.g. "Monday <br> Feb 9"
    if "<br>" in date_str:
        parts = date_str.split("<br>")
        if len(parts) < 2:
            return False
        month_day = parts[1].strip().replace(",", "").split()
        if len(month_day) < 2:
            return False
        month_token, day_token = month_day[0], month_day[1]
        day_digits = re.sub(r"[^0-9]", "", day_token)
        if not day_digits:
            return False
        day_val = int(day_digits)
        if not 1 <= day_val <= 31:
            return False
        try:
            datetime.strptime(month_token, "%b")
            return True
        except ValueError:
            try:
                datetime.strptime(month_token, "%B")
                return True
            except ValueError:
                return False
    return False


def check_rowspan_consistency(yaml_data: dict) -> bool:
    """Verify grouped assignment entries use a positive due_date_rowspan when due_date exists."""
    groups = yaml_data.get("assignment_groups", [])
    if not isinstance(groups, list):
        return False
    for group in groups:
        assignments = group.get("assignments", [])
        if not isinstance(assignments, list):
            return False
        for item in assignments:
            if not isinstance(item, dict):
                return False
            has_due = _is_non_empty_string(item.get("due_date"))
            rowspan = item.get("due_date_rowspan")
            if has_due and rowspan is not None:
                if not isinstance(rowspan, int) or rowspan <= 0:
                    return False
    return True


def validate_yaml_schema(filepath: str) -> bool:
    """Validate YAML file against expected schema."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return False

    name = os.path.basename(filepath)
    if name == "lectures.yaml":
        lectures = data.get("lectures")
        if not isinstance(lectures, list):
            return False
        for lec in lectures:
            if not isinstance(lec, dict):
                return False
            if "number" not in lec:
                return False
            if not check_date_format(str(lec.get("date", ""))):
                return False
            if "topics" not in lec or not isinstance(lec.get("topics"), list):
                return False
            slides = lec.get("slides_videos", [])
            if not isinstance(slides, list):
                return False
            for item in slides:
                if not _is_link_item(item):
                    return False
            additional = lec.get("additional_materials", [])
            if not isinstance(additional, list):
                return False
            for item in additional:
                if not _is_link_item(item):
                    return False
        return True

    if name == "assignments.yaml":
        groups = data.get("assignment_groups")
        if not isinstance(groups, list):
            return False
        for group in groups:
            if not isinstance(group, dict):
                return False
            if not _is_non_empty_string(group.get("release_date")):
                return False
            assignments = group.get("assignments")
            if not isinstance(assignments, list):
                return False
            for a in assignments:
                if not isinstance(a, dict):
                    return False
                if not _is_non_empty_string(a.get("name")):
                    return False
                materials = a.get("materials", [])
                if not isinstance(materials, list):
                    return False
                for item in materials:
                    if not _is_link_item(item):
                        return False
        return check_rowspan_consistency(data)

    if name == "recitations.yaml":
        for top_key in ("recitations_0", "recitations"):
            if top_key in data and not isinstance(data[top_key], list):
                return False
        return True

    return False


def run_all_checks(directory: str) -> Dict[str, List[str]]:
    """Run full validation suite; returns {file: [errors]}."""
    errors: Dict[str, List[str]] = {}
    files = ["lectures.yaml", "assignments.yaml", "recitations.yaml"]

    for filename in files:
        filepath = os.path.join(directory, filename)
        file_errors: List[str] = []
        if not os.path.isfile(filepath):
            file_errors.append("Missing file")
        else:
            try:
                if not validate_yaml_schema(filepath):
                    file_errors.append("Schema validation failed")
            except Exception as exc:
                file_errors.append(f"Validation exception: {exc}")
        if file_errors:
            errors[filename] = file_errors
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate S26 table YAML files")
    parser.add_argument("--directory", default=SCRIPT_DIR, help="Directory containing table YAML files")
    args = parser.parse_args()

    errors = run_all_checks(args.directory)
    if errors:
        print("[ERROR] YAML validation failed")
        for file_name, errs in errors.items():
            for err in errs:
                print(f" - {file_name}: {err}")
        return 1
    print("[INFO] YAML validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
