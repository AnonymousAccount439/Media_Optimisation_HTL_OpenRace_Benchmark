"use strict";

let chartInstance = null;
let playgroundData = null;
let renderDebounceTimer = null;

const els = {
  version: document.getElementById("versionSelect"),
  batch: document.getElementById("batchInput"),
  hiddenFrac: document.getElementById("hiddenFracSelect"),
  difficulty: document.getElementById("difficultySelect"),
  dataset: document.getElementById("datasetSelect"),
  optimizers: document.getElementById("optimizersContainer"),
  toggleAll: document.getElementById("toggleAllBtn"),
  generate: document.getElementById("generateBtn"),
  reset: document.getElementById("resetBtn"),
  canvas: document.getElementById("resultsChart"),
  title: document.getElementById("chartTitle"),
  note: document.getElementById("chartNote"),
  configBadge: document.getElementById("configBadge"),
};

// ─── Optimizer categories (mirrors Fig. 1 in the paper) ──────────────────────
// Three groups ordered classical → heuristic → surrogate so the chart layout
// makes the surrogate-model advantage over traditional methods immediately visible.
const OPTIMIZER_CATEGORIES = [
  {
    name: "Classical Design Approaches (DOE)",
    description: "Non-adaptive, fixed experimental plans",
    color: "#f59e0b",
    optimizers: [
      "FULL_FACTORIAL", "CENTRAL_COMPOSITE", "FRACTIONAL_FACTORIAL",
      "BOX_BEHNKEN", "LATIN_HYPERCUBE", "PLACKETT_BURMAN", "D_OPTIMAL",
    ],
  },
  {
    name: "Direct Heuristic Search",
    description: "No surrogate model; random or evolutionary strategies",
    color: "#ef4444",
    optimizers: ["RANDOM", "DE_DIRECT"],
  },
  {
    name: "Surrogate-Assisted Optimization",
    description: "Adaptive methods using a fitted surrogate model (GP or ANN)",
    color: "#3b82f6",
    optimizers: [
      "BO_GP_EI", "SBO_GP_PV", "SBO_GP_EI", "SBO_GP_EI_TRUNCDE",
      "SMART_BO", "SBO_ANN_PV",
    ],
  },
];

// Build a flat lookup: optimizerName → sort key (for ordering chart output)
const OPTIMIZER_CATEGORY_INDEX = {};
OPTIMIZER_CATEGORIES.forEach((cat, catIdx) => {
  cat.optimizers.forEach((opt, optIdx) => {
    OPTIMIZER_CATEGORY_INDEX[opt] = catIdx * 100 + optIdx;
  });
});

// Build a lookup: optimizerName → category object
const OPTIMIZER_TO_CATEGORY = {};
OPTIMIZER_CATEGORIES.forEach((cat) => {
  cat.optimizers.forEach((opt) => {
    OPTIMIZER_TO_CATEGORY[opt] = cat;
  });
});

// ─── Utility helpers ──────────────────────────────────────────────────────────

function navigateToPlayground(targetVersion, shouldScroll = true) {
  if (targetVersion === "hide" || targetVersion === "open") {
    els.version.value = targetVersion;
    updateChartTitle();
    renderChart();
  }
  if (shouldScroll) {
    const section = document.getElementById("playground");
    if (section && section.scrollIntoView) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
}

function handleHashNavigation(shouldScroll = false) {
  const h = (location.hash || "").toLowerCase();
  if (h === "#playground-hide") {
    navigateToPlayground("hide", shouldScroll);
  } else if (h === "#playground-open") {
    navigateToPlayground("open", shouldScroll);
  }
}

function stringHash(str) {
  let hash = 2166136261;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) || 1;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededRandBetween(rng, min, max) {
  return min + (max - min) * rng();
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

// Return a color for an optimizer, consistent with its category color.
// Within a category, lightness is varied slightly so individual lines are distinct.
function getOptimizerColor(optimizerName, allSelected) {
  const cat = OPTIMIZER_TO_CATEGORY[optimizerName];
  const hex = cat ? cat.color : "#64748b";
  const { r, g, b } = hexToRgb(hex);

  // How many from the same category are in the selection?
  const sameCategory = allSelected.filter(n => (OPTIMIZER_TO_CATEGORY[n] || {}).color === hex);
  const posInCat = sameCategory.indexOf(optimizerName);
  const total = sameCategory.length;

  // Vary lightness: base ~55%, spread by position within category
  const lightnessBase = 45;
  const lightnessStep = total > 1 ? 20 / (total - 1) : 0;
  const lightness = lightnessBase + posInCat * lightnessStep;

  // Convert hex to hue for hsl
  const max = Math.max(r / 255, g / 255, b / 255);
  const min = Math.min(r / 255, g / 255, b / 255);
  const d = max - min;
  let hue = 0;
  if (d > 0) {
    if (max === r / 255) hue = ((g / 255 - b / 255) / d) % 6;
    else if (max === g / 255) hue = (b / 255 - r / 255) / d + 2;
    else hue = (r / 255 - g / 255) / d + 4;
    hue = Math.round(hue * 60);
    if (hue < 0) hue += 360;
  }

  return `hsl(${hue} 65% ${lightness}%)`;
}

function getCategoryColor(optimizerName, opacity = 1) {
  const cat = OPTIMIZER_TO_CATEGORY[optimizerName];
  const hex = cat ? cat.color : "#64748b";
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${opacity})`;
}

// ─── Optimizer list population ────────────────────────────────────────────────

function getAvailableOptimizers() {
  if (!playgroundData) return [];
  const version = els.version.value;
  const batch = els.batch.value;
  const hiddenFrac = els.hiddenFrac.value;
  const difficulty = els.difficulty.value;
  const raceType = version === "hide" ? "Hide_The_Label" : "Open_Race";
  try {
    const modeData = playgroundData[hiddenFrac]?.[difficulty]?.[raceType]?.[batch];
    if (!modeData) return [];
    const optimizerSet = new Set();
    for (const dataset in modeData) {
      Object.keys(modeData[dataset]).forEach(opt => optimizerSet.add(opt));
    }
    return Array.from(optimizerSet);
  } catch (e) {
    console.error("Error getting available optimizers:", e);
    return [];
  }
}

function populateOptimizers() {
  const available = new Set(getAvailableOptimizers());
  els.optimizers.innerHTML = "";

  if (available.size === 0) {
    els.optimizers.innerHTML =
      '<p style="color: var(--muted); font-style: italic; grid-column: 1/-1;">No optimizers available for current selection. Try different parameters.</p>';
    return;
  }

  let globalIdx = 0;
  const assigned = new Set();

  OPTIMIZER_CATEGORIES.forEach((cat) => {
    const catOpts = cat.optimizers.filter(o => available.has(o));
    if (catOpts.length === 0) return;

    // Full-width category header
    const header = document.createElement("div");
    header.className = "opt-category-header";
    header.innerHTML = `
      <span class="opt-category-dot" style="background:${cat.color}"></span>
      <span class="opt-category-name">${cat.name}</span>
      <span class="opt-category-desc">${cat.description}</span>
    `;
    els.optimizers.appendChild(header);

    catOpts.forEach((name) => {
      assigned.add(name);
      const id = `opt_${globalIdx++}`;
      const wrapper = document.createElement("label");
      wrapper.className = "opt-item";
      wrapper.setAttribute("for", id);
      wrapper.style.setProperty("--opt-color", cat.color);

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = id;
      checkbox.value = name;
      checkbox.checked = true; // All selected by default so category trends are visible
      checkbox.addEventListener("change", debouncedRender);

      const dot = document.createElement("span");
      dot.className = "opt-color-dot";
      dot.style.background = cat.color;

      const text = document.createElement("span");
      text.textContent = name.replace(/_/g, ' ');

      wrapper.appendChild(checkbox);
      wrapper.appendChild(dot);
      wrapper.appendChild(text);
      els.optimizers.appendChild(wrapper);
    });
  });

  // Anything not in a known category → "Other"
  const uncategorized = Array.from(available).filter(o => !assigned.has(o)).sort();
  if (uncategorized.length > 0) {
    const header = document.createElement("div");
    header.className = "opt-category-header";
    header.innerHTML = `
      <span class="opt-category-dot" style="background:#64748b"></span>
      <span class="opt-category-name">Other</span>
      <span class="opt-category-desc">Uncategorized optimizers</span>
    `;
    els.optimizers.appendChild(header);

    uncategorized.forEach((name) => {
      const id = `opt_${globalIdx++}`;
      const wrapper = document.createElement("label");
      wrapper.className = "opt-item";
      wrapper.setAttribute("for", id);

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = id;
      checkbox.value = name;
      checkbox.checked = true;
      checkbox.addEventListener("change", debouncedRender);

      const text = document.createElement("span");
      text.textContent = name.replace(/_/g, ' ');

      wrapper.appendChild(checkbox);
      wrapper.appendChild(text);
      els.optimizers.appendChild(wrapper);
    });
  }
}

function getSelectedOptimizers() {
  const checked = els.optimizers.querySelectorAll('input[type="checkbox"]:checked');
  return Array.from(checked).map((c) => c.value);
}

// Return selected optimizers sorted by category order (Bayesian-GP first,
// then ANN-based, DOE, sampling, direct search) so chart layout mirrors Fig. 1.
function getOrderedSelectedOptimizers() {
  const selected = getSelectedOptimizers();
  return selected.slice().sort((a, b) => {
    const ia = OPTIMIZER_CATEGORY_INDEX[a] ?? 9999;
    const ib = OPTIMIZER_CATEGORY_INDEX[b] ?? 9999;
    return ia - ib;
  });
}

function toggleAllOptimizers() {
  const checkboxes = els.optimizers.querySelectorAll('input[type="checkbox"]');
  if (checkboxes.length === 0) return;
  const anyUnchecked = Array.from(checkboxes).some(cb => !cb.checked);
  checkboxes.forEach(cb => { cb.checked = anyUnchecked; });
  debouncedRender();
}

// ─── Chart title & config badge ───────────────────────────────────────────────

function updateChartTitle() {
  const isHide = els.version.value === "hide";
  const difficulty = els.difficulty.value;
  const hiddenFrac = parseFloat(els.hiddenFrac.value);
  const batch = els.batch.value;

  els.title.textContent = isHide
    ? `Hide-the-Label — ${difficulty} Mode (${hiddenFrac * 100}% Hidden, Batch ${batch})`
    : `Open Race — ${difficulty} Mode (Batch ${batch})`;
}

function updateConfigBadge() {
  if (!els.configBadge) return;

  const isHide = els.version.value === "hide";
  const difficulty = els.difficulty.value;
  const hiddenFrac = parseFloat(els.hiddenFrac.value);
  const batch = parseInt(els.batch.value, 10);
  const dataset = els.dataset.value;

  const datasetLabel = {
    all: "All Datasets (averaged)",
    T_Cell: "T Cell",
    TF_Cell: "TF Cell",
    Hela_regular: "HeLa Regular",
    Hela_timesaving: "HeLa Timesaving",
    rat_myocyte: "Rat Myocyte",
  }[dataset] || dataset;

  const surrogateDesc = difficulty === "Hard"
    ? "Random Forest + homoscedastic noise (σ = 0.1)"
    : "Gaussian Process (GP), low noise";
  const metricDesc = isHide
    ? "Average steps to reach target (lower = better)"
    : "Best value found so far at each step (higher = better)";
  const setupDesc = isHide
    ? `${(hiddenFrac * 100).toFixed(0)}% of pool labels hidden; ${((1 - hiddenFrac) * 100).toFixed(0)}% revealed initially`
    : "S = 200 pool evaluations per competition; R = 5 rounds";

  els.configBadge.innerHTML = `
    <div class="config-badge-title">Benchmark Configuration</div>
    <div class="config-badge-grid">
      <div class="config-badge-item">
        <span class="config-badge-label">Protocol</span>
        <span class="config-badge-value">${isHide ? "Hide-the-Label" : "Open Race"}</span>
      </div>
      <div class="config-badge-item">
        <span class="config-badge-label">Mode</span>
        <span class="config-badge-value config-mode-${difficulty.toLowerCase()}">${difficulty}</span>
      </div>
      <div class="config-badge-item">
        <span class="config-badge-label">Surrogate / Function</span>
        <span class="config-badge-value">${surrogateDesc}</span>
      </div>
      <div class="config-badge-item">
        <span class="config-badge-label">Batch Size</span>
        <span class="config-badge-value">B = ${batch}</span>
      </div>
      ${isHide ? `<div class="config-badge-item">
        <span class="config-badge-label">Hidden Fraction</span>
        <span class="config-badge-value">${(hiddenFrac * 100).toFixed(0)}% hidden</span>
      </div>` : ""}
      <div class="config-badge-item">
        <span class="config-badge-label">Dataset</span>
        <span class="config-badge-value">${datasetLabel}</span>
      </div>
      <div class="config-badge-item config-badge-full">
        <span class="config-badge-label">Setup</span>
        <span class="config-badge-value">${setupDesc}</span>
      </div>
      <div class="config-badge-item config-badge-full">
        <span class="config-badge-label">Metric</span>
        <span class="config-badge-value">${metricDesc}</span>
      </div>
    </div>
  `;
}

// ─── Data availability check ──────────────────────────────────────────────────

function hasDataForSelection() {
  if (!playgroundData) return false;
  const version = els.version.value;
  const batch = els.batch.value;
  const hiddenFrac = els.hiddenFrac.value;
  const difficulty = els.difficulty.value;
  const raceType = version === "hide" ? "Hide_The_Label" : "Open_Race";
  try {
    const data = playgroundData[hiddenFrac]?.[difficulty]?.[raceType]?.[batch];
    return data && Object.keys(data).length > 0;
  } catch (e) {
    return false;
  }
}

// ─── Placeholder data generators ─────────────────────────────────────────────

function generatePlaceholderBarData(selected, seedBase) {
  const colors = selected.map(n => getCategoryColor(n, 0.7));
  const rng = mulberry32(seedBase);
  const base = seededRandBetween(rng, 20, 50);
  const data = selected.map(() => Math.max(5, base + seededRandBetween(rng, -15, 15)));
  return {
    labels: selected,
    datasets: [{
      label: "Average Steps to Target (Placeholder)",
      data,
      backgroundColor: colors,
      borderColor: colors.map(c => c.replace(/[\d.]+\)$/, "1)")),
      borderWidth: 1,
    }],
  };
}

function generatePlaceholderLineData(selected, seedBase) {
  const steps = 30;
  const labels = Array.from({ length: steps }, (_, i) => `${i + 1}`);
  const datasets = selected.map((name) => {
    const rng = mulberry32(stringHash(`${seedBase}:${name}`));
    let level = seededRandBetween(rng, 40, 65);
    const drift = seededRandBetween(rng, 0.6, 1.6);
    const noiseAmp = seededRandBetween(rng, 0.8, 2.5);
    const data = [];
    for (let t = 0; t < steps; t++) {
      level += drift + (rng() - 0.5) * noiseAmp;
      data.push(Math.max(0, Math.min(100, level)));
    }
    for (let i = 1; i < data.length; i++) {
      if (data[i] < data[i - 1]) data[i] = data[i - 1];
    }
    const color = getCategoryColor(name, 1);
    return { label: name.replace(/_/g, ' '), data, borderColor: color, backgroundColor: color, fill: false, tension: 0.2, pointRadius: 1.8 };
  });
  return { labels, datasets };
}

// ─── Real data generators ─────────────────────────────────────────────────────

function generateBarData(selected) {
  const batch = els.batch.value;
  const hiddenFrac = els.hiddenFrac.value;
  const difficulty = els.difficulty.value;
  const selectedDataset = els.dataset.value;

  if (!playgroundData || !hasDataForSelection()) {
    const seedBase = stringHash(`hide:${selected.join(",")}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data — no results available for this configuration";
    return generatePlaceholderBarData(selected, seedBase);
  }

  try {
    const modeData = playgroundData[hiddenFrac][difficulty]["Hide_The_Label"][batch];
    const items = selected.map(opt => {
      let values = [];
      if (selectedDataset === "all") {
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData && optData.mean_steps !== undefined) values.push(optData.mean_steps);
        }
      } else {
        const optData = modeData[selectedDataset]?.[opt];
        if (optData && optData.mean_steps !== undefined) values.push(optData.mean_steps);
      }
      if (values.length === 0) return null;
      return { opt, val: values.reduce((a, b) => a + b, 0) / values.length };
    }).filter(Boolean);

    const labels = items.map(d => d.opt.replace(/_/g, ' '));
    const vals = items.map(d => d.val);
    const bgColors = items.map(d => getCategoryColor(d.opt, 0.72));
    const borderColors = items.map(d => getCategoryColor(d.opt, 1));

    els.note.textContent = "Real benchmark results — lower steps = faster convergence to target";

    return {
      labels,
      datasets: [{
        label: "Average Steps to Target (lower is better)",
        data: vals,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
      }],
    };
  } catch (e) {
    console.error("Error generating bar data:", e);
    const seedBase = stringHash(`hide:${selected.join(",")}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data — error loading results";
    return generatePlaceholderBarData(selected, seedBase);
  }
}

function generateLineData(selected) {
  const batch = els.batch.value;
  const hiddenFrac = els.hiddenFrac.value;
  const difficulty = els.difficulty.value;
  const selectedDataset = els.dataset.value;

  if (!playgroundData || !hasDataForSelection()) {
    const seedBase = stringHash(`open:${selected.join(",")}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data — no results available for this configuration";
    return generatePlaceholderLineData(selected, seedBase);
  }

  try {
    const modeData = playgroundData[hiddenFrac][difficulty]["Open_Race"][batch];
    let maxSteps = 0;

    selected.forEach(opt => {
      if (selectedDataset === "all") {
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData?.steps) maxSteps = Math.max(maxSteps, optData.steps.length);
        }
      } else {
        const optData = modeData[selectedDataset]?.[opt];
        if (optData?.steps) maxSteps = Math.max(maxSteps, optData.steps.length);
      }
    });

    const labels = Array.from({ length: maxSteps }, (_, i) => `${i}`);
    const datasets = [];

    selected.forEach((opt) => {
      let optValues = [];
      if (selectedDataset === "all") {
        const datasetValues = [];
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData?.values) datasetValues.push(optData.values);
        }
        if (datasetValues.length > 0) {
          for (let step = 0; step < maxSteps; step++) {
            const stepVals = datasetValues
              .map(vals => step < vals.length ? vals[step] : null)
              .filter(v => v !== null);
            if (stepVals.length > 0)
              optValues.push(stepVals.reduce((a, b) => a + b, 0) / stepVals.length);
          }
        }
      } else {
        const optData = modeData[selectedDataset]?.[opt];
        if (optData?.values) optValues = optData.values;
      }

      if (optValues.length > 0) {
        for (let i = 1; i < optValues.length; i++) {
          if (optValues[i] < optValues[i - 1]) optValues[i] = optValues[i - 1];
        }
        const color = getOptimizerColor(opt, selected);
        datasets.push({
          label: opt.replace(/_/g, ' '),
          data: optValues,
          borderColor: color,
          backgroundColor: color,
          fill: false,
          tension: 0.2,
          pointRadius: 1.8,
        });
      }
    });

    if (datasets.length === 0) {
      const seedBase = stringHash(`open:${selected.join(",")}:${batch}:${hiddenFrac}`);
      els.note.textContent = "Showing placeholder data — no results for selected optimizers";
      return generatePlaceholderLineData(selected, seedBase);
    }

    els.note.textContent = "Real benchmark results — higher best-value = better performance";
    return { labels, datasets };
  } catch (e) {
    console.error("Error generating line data:", e);
    const seedBase = stringHash(`open:${selected.join(",")}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data — error loading results";
    return generatePlaceholderLineData(selected, seedBase);
  }
}

// ─── Chart rendering ──────────────────────────────────────────────────────────

function renderChart() {
  const version = els.version.value;
  const selected = getOrderedSelectedOptimizers(); // sorted by category

  if (selected.length === 0) {
    if (els.note) els.note.textContent = "Select at least one optimizer to display results.";
    if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
    updateConfigBadge();
    return;
  }

  const ctx = els.canvas.getContext("2d");
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  if (version === "hide") {
    const data = generateBarData(selected);
    chartInstance = new Chart(ctx, {
      type: "bar",
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)} steps` },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: "Average Steps to Target (Lower = Better)" },
          },
          x: { title: { display: true, text: "Optimizer (grouped: DOE → Direct Search → Surrogate-Assisted)" } },
        },
      },
    });
  } else {
    const data = generateLineData(selected);
    chartInstance = new Chart(ctx, {
      type: "line",
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: {
            display: true,
            position: "right",
            labels: { boxWidth: 12, font: { size: 12 } },
          },
          tooltip: { enabled: true },
        },
        scales: {
          y: { title: { display: true, text: "Best Value Found So Far (Higher = Better)" } },
          x: { title: { display: true, text: "Evaluation Step" } },
        },
        elements: { point: { radius: 2 }, line: { tension: 0.1 } },
      },
    });
  }

  updateChartTitle();
  updateConfigBadge();
}

// Debounce rapid checkbox toggles to avoid excessive re-renders
function debouncedRender() {
  clearTimeout(renderDebounceTimer);
  renderDebounceTimer = setTimeout(renderChart, 250);
}

// ─── Form helpers ─────────────────────────────────────────────────────────────

function resetForm() {
  els.version.value = "hide";
  els.batch.value = "1";
  els.hiddenFrac.value = "0.95";
  els.difficulty.value = "Regular";
  els.dataset.value = "all";
  populateOptimizers();
  updateChartTitle();
  renderChart();
}

// Comment 2.2 – auto-render: any top-level parameter change immediately
// re-renders the chart without requiring the "Generate Results" button.
function onParameterChange() {
  populateOptimizers();
  updateChartTitle();
  renderChart();
}

function onDatasetChange() {
  updateChartTitle();
  renderChart();
}

// ─── Data loading ─────────────────────────────────────────────────────────────

async function loadRealData() {
  if (window.playgroundData) {
    playgroundData = window.playgroundData;
    console.log("Loaded playground data from embedded script");
    console.log("Available configurations:", Object.keys(playgroundData));
    return;
  }
  try {
    const response = await fetch("playground_data.json");
    if (response.ok) {
      playgroundData = await response.json();
      console.log("Loaded playground data from JSON file");
    } else {
      console.warn("Could not load playground data, using placeholder data");
    }
  } catch (e) {
    console.warn("Error loading playground data:", e);
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  await loadRealData();

  populateOptimizers();
  updateChartTitle();

  // All top-level parameter controls auto-render on change (comment 2.2)
  els.version.addEventListener("change", onParameterChange);
  els.batch.addEventListener("change", onParameterChange);
  els.hiddenFrac.addEventListener("change", onParameterChange);
  els.difficulty.addEventListener("change", onParameterChange);
  els.dataset.addEventListener("change", onDatasetChange);

  els.toggleAll.addEventListener("click", toggleAllOptimizers);
  els.generate.addEventListener("click", renderChart);
  els.reset.addEventListener("click", resetForm);

  // URL ?target= deep-link support
  const urlParams = new URLSearchParams(window.location.search);
  const target = urlParams.get("target");
  if (target === "hide" || target === "open") {
    els.version.value = target;
    populateOptimizers();
    updateChartTitle();
  }

  window.addEventListener("hashchange", () => handleHashNavigation(true));

  // Initial render
  renderChart();
  handleHashNavigation(false);
}

document.addEventListener("DOMContentLoaded", init);
