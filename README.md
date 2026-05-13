# Website Notes

Quick notes for course staff maintaining the IDL website.


## Transitioning between semesters

1. Create a new semester folder from the most recent compatible template.
2. Update root `index.html` to point to the new semester (for example `url=./F26/index.html`).
3. Add the semester switcher links in nav/header where applicable.
4. Remove old semester-specific content that should not carry over.
5. Keep only template placeholders unless real semester content is confirmed.


## Organization

- `FXX` / `SXX`: semester sites.
- `index.html`: redirect to current semester.
- `shared/`: pages/assets shared across semesters.
- `exp/`: optional sandbox area for temporary experiments.


## F26 Notes

- `F26/` is set up as a template-first semester folder.
- `F26/documents/` is scaffolded (not preloaded with full S26 material).
- Slide/content ingestion is designed to be maintained via automation:
  - Discord bot intake queue under `tools/f26_discord_bot/`
  - Workflow processing on Tuesday/Thursday


## Pseudocode Page

http://deeplearning.cs.cmu.edu/F20/pseudocode.html  
Historical work-in-progress reference. Keep this section updated only if the team actively maintains it.


## Technologies

- Mix of static HTML/CSS + table generation scripts (semester-specific).
- Semester table pages are generated from YAML data using Jinja.
- Keep dependencies lightweight unless a strong need comes up.


## Misc

- Keep semester content authoritative and traceable to course staff sources.
- Favor automation for repetitive updates (slides/tables) and manual review for exceptions.
- Keep PRs small and reviewable when possible.
