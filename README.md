# CMU-IDeeL Course Website

Static website repo for 11-785/11-485 (Introduction to Deep Learning).

## Current semester

- Root [index.html](./index.html) redirects to the active semester.
- As of now, it points to `S26`.

## Repository layout

- `S20`..`S26`, `F20`..`F25`: semester websites.
- `shared/`: assets and pages used across semesters.
- `.github/workflows/`: automation workflows.

## Semester rollover

1. Copy the latest semester directory (example: `S26 -> F26`).
2. Update semester labels/links in desktop and mobile headers.
3. Validate pages locally.
4. Update root `index.html` redirect only after validation.

## S26 table pipeline

S26 tables are generated from YAML:

- Source data: `S26/pages/tables_data/*.yaml`
- Templates: `S26/pages/tables_templates/*.html`
- Generated output: `S26/pages/tables/*.html`

Run from repo root:

```bash
python S26/pages/tables_data/validate_tables_data.py
python S26/pages/tables_data/upload_slides.py --dry-run
python S26/pages/tables_data/generate_table.py
python S26/pages/tables_data/preflight_f26.py --skip-external
```

## Automation

- Workflow: `.github/workflows/s26_auto_slide_upload.yml`
- Scheduled + manual trigger
- Runs validation, slide-link update, table generation, and preflight checks before commit.
