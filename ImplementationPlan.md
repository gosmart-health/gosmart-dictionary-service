# Implementation Plan: Web Portal Dark Mode Support

## Goal

Add first-class **Dark Mode** support across every page of the `/web` portal (`/`, `/login`, `/register`, `/verify-email`, `/keys`, `/chat`, `/explore`) with zero layout changes and minimal footprint.

Because the portal uses a single self-contained stylesheet ([`src/gosmart_dictionary_service/web/static/style.css`](./src/gosmart_dictionary_service/web/static/style.css)) already organized around `:root` CSS custom properties—and **zero** templates hardcode inline colors—dark mode can be added cleanly in just **3 frontend files** plus unit test coverage.

---

## Architecture & Theme Strategy

### Recommended Approach: System Preference (`prefers-color-scheme`) + Topbar Toggle (`localStorage`)

1. **Semantic CSS Variables (`style.css`)**:
   - Promote the remaining ~10 hardcoded light hex values (`#fff` on inputs/selects/buttons, `#e0f2fe`/`#075985` on `.pill-info`, `#eef4ff`/`#c7d7fe` on `.citation-rel-chip` / `.chat-prompt-chip:hover` / `.mcp-agent-tab:hover`, and `#0f172a`/`#e2e8f0` on `.secret-box` & `pre.code-block`) into semantic `:root` tokens:
     - `--input-bg`, `--info-bg`, `--info-text`, `--chip-bg`, `--chip-border`, `--chip-hover-bg`, `--code-bg`, `--code-text`, `--code-border`, `--focus-ring`.
   - Define a single dark palette block targeting both:
     - `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ... } }` (automatic OS dark mode by default)
     - `:root[data-theme="dark"] { ... }` (explicit user override)
   - Set `color-scheme: light` / `color-scheme: dark` on `:root` so native `<select>` dropdowns, scrollbars, and inputs match the active theme automatically.
2. **Zero-Flash (No FOUC) Boot Script (`base.html`)**:
   - Place a 4-line synchronous inline script in `<head>` (before `<link rel="stylesheet">`) that reads `localStorage.getItem("gosmart_ds_theme")` (`"light"` | `"dark"`) and sets `document.documentElement.setAttribute("data-theme", ...)` before first paint.
3. **Compact Topbar Theme Toggle (`base.html` + `portal.js`)**:
   - Add an accessible `<button id="theme-toggle-btn" class="btn btn-secondary btn-sm theme-toggle">` in the topbar `<nav>` so users can switch between **Light** and **Dark** at any time (works for both signed-in and logged-out pages; persists across reloads via `localStorage`).
   - Falls back gracefully to OS `prefers-color-scheme` when JavaScript is disabled.

### Alternative (Even Simpler — Zero JS)

- **CSS-Only Automatic Dark Mode**: Only update `style.css` with `@media (prefers-color-scheme: dark)` and `color-scheme: light dark`. Follows macOS/Windows/browser appearance automatically with **1 file changed**, at the cost of no manual in-page toggle.

---

## Proposed Dark Palette Preview

| Token | Light Mode (Current) | Dark Mode (Proposed) |
| :--- | :--- | :--- |
| `--bg` | `#f6f8fb` | `#0b1220` |
| `--surface` | `#ffffff` | `#111a2c` |
| `--surface-muted` | `#f0f4f8` | `#182338` |
| `--input-bg` | `#ffffff` | `#0f172a` |
| `--border` | `#d8e1eb` | `#2a3954` |
| `--text` | `#162231` | `#e5edf8` |
| `--text-muted` | `#526377` | `#9fb0c7` |
| `--primary` / `--primary-hover` | `#0f62fe` / `#0043ce` | `#4d8dff` / `#73a6ff` |
| `--code-bg` / `--code-text` | `#0f172a` / `#e2e8f0` | `#070d18` / `#e2e8f0` *(with subtle border)* |
| `--success-bg` / `--success-border` / `--success-text` | `#e7f6ec` / `#86efac` / `#14532d` | `#0d2d1d` / `#1f6f47` / `#86efac` |
| `--warn-bg` / `--warn-border` / `--warn-text` | `#fff8e6` / `#fcd34d` / `#78350f` | `#33250a` / `#8a6116` / `#fde68a` |
| `--danger-bg` / `--danger-border` / `--danger-text` | `#fef2f2` / `#fca5a5` / `#991b1b` | `#361318` / `#8f2d36` / `#fca5a5` |
| `--info-bg` / `--info-text` | `#e0f2fe` / `#075985` | `#0e2a42` / `#7dd3fc` |
| `--chip-bg` / `--chip-border` | `#eef4ff` / `#c7d7fe` | `#18294a` / `#2f4f88` |

---

## Detailed Implementation Steps

1. **[`src/gosmart_dictionary_service/web/static/style.css`](./src/gosmart_dictionary_service/web/static/style.css)**:
   - Replace hardcoded `#fff`, `#e0f2fe`, `#075985`, `#eef4ff`, `#c7d7fe`, `#0f172a`, and `#e2e8f0` references with the new semantic CSS variables.
   - Add dark mode variable overrides for both `:root[data-theme="dark"]` and `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ... } }`, including `color-scheme` and `.theme-toggle` button styling.
2. **[`src/gosmart_dictionary_service/web/templates/base.html`](./src/gosmart_dictionary_service/web/templates/base.html)**:
   - Add `<meta name="color-scheme" content="light dark">` and the tiny pre-paint `data-theme` initializer script in `<head>`.
   - Add the compact **Light / Dark** toggle button (`id="theme-toggle-btn"`) to `.nav-links`.
3. **[`src/gosmart_dictionary_service/web/static/portal.js`](./src/gosmart_dictionary_service/web/static/portal.js)**:
   - Add `initThemeToggle()` to read/toggle `data-theme`, persist to `localStorage` (`gosmart_ds_theme`), sync button label/ARIA state (`aria-pressed`), and react to OS `prefers-color-scheme` changes when no explicit override is saved.
4. **Tests & Documentation**:
   - Add a modular unit test in [`tests/unit/test_web_portal.py`](./tests/unit/test_web_portal.py) verifying the dark-mode CSS custom properties (`prefers-color-scheme: dark` and `[data-theme="dark"]`), `<meta name="color-scheme">`, `#theme-toggle-btn` markup, and `initThemeToggle` JS hook.
   - Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest tests/unit`, and PyCharm `lint_files`.
