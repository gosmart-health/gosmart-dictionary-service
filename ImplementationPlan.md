# Implementation Plan: Migrate Architecture Design Note to Zensical Site

## Goal

Publish `docs/design_notes/architecture.md` (445 lines) as a **single page** in the
Zensical documentation site at `docs/zensical/docs/architecture.md`, expose it in the
site navigation, and **keep** the original design note untouched. Content migrates
verbatim (no editorial rewrite).

## Confirmed Decisions

| Decision | Choice |
| :--- | :--- |
| Page layout | Single page `architecture.md` (not a split section) |
| Original `docs/design_notes/architecture.md` | **Kept** (no deletion) |
| Nav placement | Top-level "Architecture" entry between "Getting Started" and "About" |

## Findings

- **Site conventions**: existing pages use front matter `title:` / `description:` plus
  an H1 (`index.md`, `getting-started/index.md`, `about/license.md`).
- **Mermaid support**: `zensical.toml` already configures
  `pymdownx.superfences.custom_fences` for `mermaid` — both diagrams in the source
  (`sequenceDiagram` in §2.2, `flowchart` in §3.1) render without config changes.
- **Extensions enabled**: `tables`, `admonition`, `toc.permalink`,
  `pymdownx.details/tabbed/highlight` — all syntax used by the source is covered.
- **Internal anchors**: the only internal links are two
  `[Section 4.2](#42-terminology-encoding--response-contract)` references (source
  lines 203, 384). They point to heading `### 4.2 Terminology Encoding & Response
  Contract`; keeping headings verbatim preserves the slugs. No relative file links
  exist, so the `--strict` build has minimal break risk.
- **Docs tooling**: `docs` dependency group = `zensical>=0.1.3`; build from
  `docs/zensical/` via `uv run --group docs zensical build --strict --clean`.
- **Ruff**: no Python changes; lint/format gate should remain green (verified anyway).

## Implementation Steps

1. Write `ImplementationPlan.md` at the repo root (this file, for copying into a
   GitHub Issue) — per AGENTS.md rule 3.
2. Create `docs/zensical/docs/architecture.md`:
   - Front matter: `title: Architecture` + `description` in the existing page style.
   - Body: verbatim copy of the source document, including the original H1
     (`# GoSmart Health Terminology Dictionary and Ontology Graph Service`), all
     numbered section headings (anchors preserved), both Mermaid diagrams, all
     tables, and the Appendix.
   - Built by concatenating front matter + the untouched source file to guarantee a
     byte-exact migration.
3. Update `docs/zensical/zensical.toml` — insert into `nav`:
   `{ "Architecture" = "architecture.md" },` between the "Getting Started" and
   "About" entries.
4. Update `docs/zensical/README.md` — extend the `Structure` tree with
   `architecture.md` (documentation-sync rule).
5. Update `docs/zensical/docs/index.md` — add an Architecture link in the
   `## Next` section for discoverability (one line).

## Validation

- `cd docs/zensical && uv run --group docs zensical build --strict --clean` → must
  pass with **no broken links/anchor warnings**; confirm
  `site/architecture/index.html` is generated and contains both Mermaid blocks.
- Verify the two `#42-terminology-encoding--response-contract` anchors resolve on
  the built page.
- `uv run ruff check .` and `uv run ruff format --check .` → pass (AGENTS.md gate).
- Confirm `docs/design_notes/architecture.md` is unchanged (kept as-is).

## Risks / Notes

- Anchor slugs: zensical's slugifier should produce the same `#42-…` IDs as the
  source's manual links; the strict build verifies this — if a slug differs, the two
  links are fixed in the new page only (source stays untouched).
- The doc is written in internal design-note voice ("Proposed Use Cases", "Why We
  Need This Service?"); it is migrated verbatim per the confirmed decision. A later
  editorial pass to soften user-facing phrasing is possible on request, but out of
  scope.
- `ImplementationPlan.md` at repo root is a staging file for the GitHub issue and can
  be excluded from the commit if preferred.

---

# Task 2: Sync Zensical Pages with Root README (README = truth)

## Goal

Align the Zensical site pages with the repository root `README.md`, which is the
source of truth.

## Changes

1. Rewrite `docs/zensical/docs/getting-started/index.md`:
   - Full `GOSMART_DS_*` configuration table (39 rows) replaces the 7-bullet
     "key defaults" list.
   - Add `gosmart-ds serve | bootstrap | ingest-dicom | ingest-clinical-terms`
     command blocks (dry-run, `--bootstrap`, `--compute-embeddings`,
     `--profile full|demo-core`).
   - Add "Training the spaCy NLU Model (`gosmart-clinical-nlu`)" and
     "Unified Search & Clinical NLU Service" sections from the README.
   - Replace the 3-bullet endpoint list with the full "Web Portal & Endpoints"
     section (portal pages, REST endpoints, `test_main.http` note).
   - Replace the stale project-layout tree with the README tree (`training/`,
     `nlu_models/`, `devop-scripts/`, `notebooks/`, `actor/`, `ingestor/`,
     `model/`, `nlu/`).
   - Rename "Testing and linting" heading to "Development" (README heading).
   - Repo-relative links rendered as inline code (relative repo links would
     break the `--strict` site build).
2. Sync `docs/zensical/docs/index.md` (site home):
   - Intro paragraph mentions the multi-page HTML web portal and Brevo email
     registration, matching the README description.
   - Stack table expanded from 4 rows to the README's 9 rows.
   - "Why" bullets kept (not contradicted by README).

## Validation

- `uv run --group docs zensical build --strict --clean` → no broken links.
- `uv run ruff check .` / `uv run ruff format --check .` → pass.
