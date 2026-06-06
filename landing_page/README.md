# Active Learning Landing Page

**Quick Start:** Double-click `index.html` — no installation needed.

## Files

| File | Size | Purpose |
|------|------|---------|
| `index.html` | 20 KB | Main overview page |
| `playground.html` | 6.7 KB | Interactive playground |
| `script.js` | 17 KB | JavaScript logic & charts |
| `style.css` | 8.7 KB | Styling |
| `playground_data.js` | 3.6 MB | Embedded benchmark data |
| `README.md` | — | This file |

## Usage

- **Overview:** Open `index.html`, read about the project, navigate to playground via links.
- **Playground:** Open `playground.html` → select parameters → click "Generate Results".
- Pages link to each other and work independently.

## Playground Parameters

Select benchmark type (Hide the Label / Open Race), optimizers, batch size, difficulty, dataset, and hidden fraction. Results render as real-time charts.

## Sharing

Zip the folder and send it. Recipients just unzip and double-click `index.html`.

## Technical

Pure HTML/CSS/JS. Data is embedded (no server needed). Charts via Chart.js (CDN; requires internet on first load, then cached). Works in all modern browsers. Total size ~3.7 MB, mostly data.

**Troubleshooting:** No charts → check internet connection or browser console (F12). Stale data → hard refresh (`Ctrl+Shift+R` / `Cmd+Shift+R`). Broken layout → ensure all 6 files are in the same folder.

## About

An interactive benchmark for active learning in cell culture media optimization. Bayesian Optimization (BO_GP_EI, SBO_GP_PV) reduces experiments by 50–90%, reaching targets in ~3–5 steps vs. 10–50+ for random sampling, across two benchmark types: *Hide the Label* (selecting from candidates) and *Open Race* (continuous optimization).
