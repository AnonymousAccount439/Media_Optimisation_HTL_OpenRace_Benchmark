# Active Learning Landing Page

## 🚀 Quick Start

**Simply double-click `index.html` to open the website!**

No installation, no server, no command line needed.

---

## 📁 What's in This Folder

This folder contains everything needed for the interactive landing page:

### Essential Files

1. **`index.html`** (20 KB)
   - Main overview page
   - Opens in your browser

2. **`playground.html`** (6.7 KB)
   - Interactive playground
   - Real-time visualizations

3. **`script.js`** (17 KB)
   - Main JavaScript logic
   - Handles all interactions and charts

4. **`style.css`** (8.7 KB)
   - All the styling
   - Makes everything look good

5. **`playground_data.js`** (3.6 MB)
   - All the benchmark data
   - Embedded directly for instant loading

6. **`README.md`** (this file)
   - Simple instructions

---

## 🎯 How to Use

### Option 1: Overview Page
- Double-click **`index.html`**
- Read about the project
- Click links to navigate to playground

### Option 2: Go Straight to Playground
- Double-click **`playground.html`**
- Select parameters
- Click "Generate Results"
- See visualizations

---

## 🌐 Navigation

- **From index.html**: Click buttons/links to go to playground
- **From playground.html**: Click "← Back to Overview" to return to index
- Both pages work independently

---

## 📊 The Playground

Interactive features:
- **Select Benchmark Type**: Hide the Label or Open Race
- **Choose Optimizers**: Compare multiple strategies
- **Set Parameters**: Batch size, difficulty, dataset, hidden fraction
- **View Results**: Real-time charts with actual benchmark data

---

## 💾 Sharing

To share with others:
1. Zip this entire folder
2. Send the zip file
3. Tell recipient: "Unzip and double-click index.html"

That's it! They don't need anything installed.

---

## 🔧 Technical Notes

### How It Works
- Pure HTML/CSS/JavaScript
- Data embedded as JavaScript (no server needed)
- Charts powered by Chart.js (loaded from CDN)
- Works offline after first load (Chart.js cached)

### Browser Compatibility
- ✅ Chrome
- ✅ Firefox
- ✅ Safari
- ✅ Edge

### File Size
- Total: ~3.7 MB
- Mostly data (3.6 MB)
- Very fast to load (local files)

---

## ❓ Troubleshooting

### Charts not showing?
- Make sure you have internet connection (for Chart.js CDN)
- Check browser console (F12) for errors

### Old data showing?
- Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)

### Page looks broken?
- Make sure all 6 files are in the same folder
- Try opening in a different browser

---

## 🎓 What This Project Does

This is an **interactive benchmark** for **active learning** in **cell culture media optimization**.

### The Challenge
Cell media formulation is expensive and time-consuming. Traditional methods (Design of Experiments) require many experiments to find optimal conditions.

### The Solution
**Bayesian Optimization** and other intelligent methods can:
- Reduce experiments by 50-90%
- Find optimal formulations faster
- Save time and money

### The Benchmarks
1. **Hide the Label**: Choose from existing candidates efficiently
2. **Open Race**: Explore continuous optimization spaces

### Key Finding
**BO_GP_EI** and **SBO_GP_PV** (Bayesian methods) consistently outperform traditional approaches, reaching targets in ~3-5 steps vs 10-50+ steps for random sampling.

---

## 📚 More Information

For the full codebase, datasets, and research details, explore the parent directory structure.

---

## 🎉 That's It!

Enjoy exploring the benchmarks!

**Made with ❤️ for the scientific community**
