import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Set, Tuple

import yaml

from validate_tables_data import run_all_checks


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PAGES_DIR, os.pardir))
GENERATED_TABLES = [
    os.path.join(PAGES_DIR, "tables", "lectures_table.html"),
    os.path.join(PAGES_DIR, "tables", "recitations.html"),
    os.path.join(PAGES_DIR, "tables", "assignments_table.html"),
]


def _collect_urls(node: Any, out: Set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "url" and isinstance(v, str):
                out.add(v.strip())
            else:
                _collect_urls(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_urls(item, out)


def _load_yaml_files() -> List[Dict[str, Any]]:
    items = []
    for name in ["lectures.yaml", "assignments.yaml", "recitations.yaml"]:
        path = os.path.join(SCRIPT_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            items.append(yaml.safe_load(f) or {})
    return items


def _resolve_local_url(url: str) -> str:
    url = urllib.parse.unquote(url.strip())
    return os.path.normpath(os.path.join(ROOT_DIR, url))


def check_links(
    timeout: int = 5, retries: int = 2, strict_external: bool = False, skip_external: bool = False
) -> bool:
    urls: Set[str] = set()
    for data in _load_yaml_files():
        _collect_urls(data, urls)

    local_missing: List[str] = []
    external_failed: List[str] = []
    external_ok = 0
    local_ok = 0

    for url in sorted(urls):
        if not url:
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in ("http", "https"):
            if skip_external:
                continue
            ok = False
            for _ in range(retries + 1):
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        code = getattr(resp, "status", 200)
                        if 200 <= code < 400:
                            ok = True
                            break
                except Exception:
                    continue
            if ok:
                external_ok += 1
            else:
                external_failed.append(url)
        else:
            full_path = _resolve_local_url(url)
            if os.path.exists(full_path):
                local_ok += 1
            else:
                local_missing.append(f"{url} -> {full_path}")

    print(f"[INFO] Link Check: local_ok={local_ok} external_ok={external_ok}")
    if local_missing:
        print("[ERROR] Missing local links:")
        for item in local_missing:
            print(f" - {item}")
    if external_failed:
        level = "[ERROR]" if strict_external else "[WARN]"
        print(f"{level} Unreachable external links:")
        for item in external_failed:
            print(f" - {item}")

    if strict_external:
        return len(local_missing) == 0 and len(external_failed) == 0
    return len(local_missing) == 0


def _run_generate_table() -> bool:
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_table.py")]
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    return result.returncode == 0


def check_generated_freshness() -> bool:
    if not _run_generate_table():
        print("[ERROR] Failed to run generate_table.py")
        return False
    cmd = ["git", "diff", "--name-only", "--"] + GENERATED_TABLES
    out = subprocess.check_output(cmd, cwd=os.path.dirname(ROOT_DIR), text=True).strip()
    if out:
        print("[ERROR] Generated tables were stale before regeneration:")
        print(out)
        return False
    print("[INFO] Generated tables are fresh")
    return True


def check_no_manual_edits_in_tables() -> bool:
    # Current enforceable proxy: tables must be exactly reproducible by generate_table.py.
    return check_generated_freshness()


def check_all(
    timeout: int = 5, retries: int = 2, strict_external: bool = False, skip_external: bool = False
) -> bool:
    validations = run_all_checks(SCRIPT_DIR)
    if validations:
        print("[ERROR] Schema validation failed:")
        for file_name, errs in validations.items():
            for err in errs:
                print(f" - {file_name}: {err}")
        schema_ok = False
    else:
        print("[INFO] Schema: 100% valid")
        schema_ok = True

    links_ok = check_links(
        timeout=timeout, retries=retries, strict_external=strict_external, skip_external=skip_external
    )
    freshness_ok = check_generated_freshness()
    tables_ok = check_no_manual_edits_in_tables()

    return schema_ok and links_ok and freshness_ok and tables_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run F26-style preflight checks on S26 pipeline")
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--strict-external", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    ok = check_all(
        timeout=args.timeout,
        retries=args.retries,
        strict_external=args.strict_external,
        skip_external=args.skip_external,
    )
    if not ok:
        print("[ERROR] Preflight failed")
        return 1
    print("[INFO] Preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
