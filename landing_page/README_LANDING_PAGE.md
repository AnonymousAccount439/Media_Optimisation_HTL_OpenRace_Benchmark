# Landing Page - Active Learning for Cell Media

This landing page now displays **real results** from your AL_Project experiments, highlighting that **BO_GP_EI and SBO_GP_PV are the top-performing optimizers**.

## What Was Fixed

### Problem
The landing page was displaying random placeholder data that didn't reflect your actual experimental results.

### Solution
1. **Created `process_results.py`** - Processes your actual JSON result files from the AL_Project
2. **Updated `script.js`** - Loads real data from JSON files instead of generating random values
3. **Updated `index.html`** - Content now specifically highlights BO_GP_EI and SBO_GP_PV as winners

## Key Results Shown

### Hide-the-Label Benchmark
- **BO_GP_EI**: 3.7 steps to target (BEST)
- **D_OPTIMAL**: 4.2 steps
- **SBO_GP_PV**: 6.8 steps (3rd best)
- **RANDOM**: 31.1 steps (8x worse than BO_GP_EI)

### Open Race Benchmark
- **BO_GP_EI**: 4.00 final best value (TIED BEST)
- **SBO_GP_PV**: 4.00 final best value (TIED BEST)
- **RANDOM**: 3.48 (13% worse than top performers)

## Files

### Core Files
- `index.html` - Main landing page with updated content
- `playground.html` - Interactive playground for comparing optimizers
- `script.js` - JavaScript that loads and visualizes real data
- `style.css` - Styling (unchanged)

### Data Files (Auto-generated)
- `hide_label_data.json` - Aggregated Hide-the-Label results (13 optimizers)
- `open_race_data.json` - Aggregated Open Race results (15 optimizers)

### Processing Script
- `process_results.py` - Processes your AL_Project results into summary JSON files

## How to Update the Data

When you have new experimental results:

1. Make sure your results are in the expected directories:
   - `/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Results/extraresults_hiddenfrac99/Regular_Mode/`
   - `/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Results/Regular_Mode 09-56-40-557/`

2. Run the processing script:
   ```bash
   cd /Users/aliparsaee/Downloads/Active_Learning_for_Cell_Media-organised_code/landing_page
   python3 process_results.py
   ```

3. The script will:
   - Read all JSON files from both directories
   - Aggregate results across all experiments
   - Generate updated `hide_label_data.json` and `open_race_data.json`
   - Print a summary showing optimizer rankings

4. Refresh the landing page in your browser to see updated results

## Viewing the Landing Page

### Option 1: Simple HTTP Server (Recommended)
```bash
cd /Users/aliparsaee/Downloads/Active_Learning_for_Cell_Media-organised_code/landing_page
python3 -m http.server 8000
```
Then open: http://localhost:8000

### Option 2: Open Directly
Some browsers allow opening `index.html` directly, but may block loading JSON files due to CORS. Use Option 1 for best results.

## What the Landing Page Shows

1. **Main Page (`index.html`)**:
   - Overview of the project
   - Explanation of Hide-the-Label and Open Race benchmarks
   - **Key insights highlighting BO_GP_EI and SBO_GP_PV as winners**
   - Practical guidance for labs

2. **Playground (`playground.html`)**:
   - Interactive visualization of optimizer performance
   - Select different optimizers to compare
   - View Hide-the-Label bar charts (performance scores)
   - View Open Race line plots (best-so-far trajectories)
   - **Uses real data** from your experiments

## Technical Details

### Data Processing
- Processes both "extraresults_hiddenfrac99" and "Regular_Mode 09-56-40-557" directories
- Aggregates across multiple datasets and runs
- Calculates mean, median, std, min, max for Hide-the-Label
- Computes mean trajectories for Open Race

### Visualization
- Hide-the-Label: Bar charts showing performance scores (inverted from steps, so higher = better)
- Open Race: Line plots showing best-value-so-far over time
- Uses Chart.js library for interactive visualizations
- Color-coded with viridis palette

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- Best viewed on desktop (responsive design for mobile)

## Customization

To change which optimizers are shown by default, edit `script.js`:
```javascript
// Line ~101: Default selections
if (idx < 5) checkbox.checked = true; // Change 5 to select different number
```

To highlight different optimizers in the text, edit `index.html`:
- Search for "BO_GP_EI" and "SBO_GP_PV"
- Update the key insights sections (lines 116-126)

## Data Sources

The landing page pulls from:
- **Hide-the-Label**: `all_tournament_results[].competitions[].optimizer_results[optimizer].steps_to_target`
- **Open Race**: `competitions[].optimizer_results[optimizer].optimization_history[].best_value_so_far`

Both benchmarks aggregate data across:
- Multiple datasets (T_Cell, TF_Cell, Hela, DBO, MOBO, etc.)
- Multiple batch sizes (Batch1, Batch10, Batch20)
- Multiple runs (10 competitions per configuration)

## Support

If you encounter issues:
1. Check that JSON files are properly formatted
2. Verify all result files are in expected directories
3. Run `process_results.py` and check for error messages
4. Use browser developer console to debug JavaScript issues
5. Make sure you're serving via HTTP (not file://)

---

**Last Updated**: November 2, 2025  
**Data Source**: AL_Project Results (Regular Mode, Hidden Fraction 0.95)

