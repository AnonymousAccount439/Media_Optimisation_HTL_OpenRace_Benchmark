"use strict";

let chartInstance = null;
let playgroundData = null;

const els = {
  version: document.getElementById("versionSelect"),
  batch: document.getElementById("batchInput"),
  hiddenFrac: document.getElementById("hiddenFracSelect"),
  difficulty: document.getElementById("difficultySelect"),
  dataset: document.getElementById("datasetSelect"),
  optimizers: document.getElementById("optimizersContainer"),
  generate: document.getElementById("generateBtn"),
  reset: document.getElementById("resetBtn"),
  canvas: document.getElementById("resultsChart"),
  title: document.getElementById("chartTitle"),
  note: document.getElementById("chartNote"),
};

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

function getColorPalette(n) {
  const colors = [];
  for (let i = 0; i < n; i++) {
    const hue = Math.round((360 / Math.max(3, n)) * i);
    colors.push(`hsl(${hue} 70% 60%)`);
  }
  return colors;
}

function getAvailableOptimizers() {
  if (!playgroundData) return [];
  
  const version = els.version.value;
  const batch = els.batch.value; // Keep as string for JSON key access
  const hiddenFrac = els.hiddenFrac.value; // Keep as string for JSON key access
  const difficulty = els.difficulty.value;
  const raceType = version === "hide" ? "Hide_The_Label" : "Open_Race";
  
  try {
    const modeData = playgroundData[hiddenFrac]?.[difficulty]?.[raceType]?.[batch];
    if (!modeData) return [];
    
    // Collect all optimizers across all datasets
    const optimizerSet = new Set();
    for (const dataset in modeData) {
      const datasetOpts = modeData[dataset];
      Object.keys(datasetOpts).forEach(opt => optimizerSet.add(opt));
    }
    
    return Array.from(optimizerSet).sort();
  } catch (e) {
    console.error("Error getting available optimizers:", e);
    return [];
  }
}

function populateOptimizers() {
  const available = getAvailableOptimizers();
  els.optimizers.innerHTML = "";
  
  if (available.length === 0) {
    els.optimizers.innerHTML = '<p style="color: var(--muted); font-style: italic;">No optimizers available for current selection. Try different parameters.</p>';
    return;
  }
  
  available.forEach((name, idx) => {
    const id = `opt_${idx}`;
    const wrapper = document.createElement("label");
    wrapper.className = "opt-item";
    wrapper.setAttribute("for", id);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = id;
    checkbox.value = name;
    // Default to selecting first 5 optimizers
    if (idx < 5) checkbox.checked = true;

    const text = document.createElement("span");
    text.textContent = name;

    wrapper.appendChild(checkbox);
    wrapper.appendChild(text);
    els.optimizers.appendChild(wrapper);
  });
}

function getSelectedOptimizers() {
  const checked = els.optimizers.querySelectorAll('input[type="checkbox"]:checked');
  return Array.from(checked).map((c) => c.value);
}

function updateChartTitle() {
  const isHide = els.version.value === "hide";
  const difficulty = els.difficulty.value;
  const hiddenFrac = parseFloat(els.hiddenFrac.value);
  const batch = els.batch.value;
  
  els.title.textContent = isHide 
    ? `Hide the Label — ${difficulty} Mode (${hiddenFrac*100}% Hidden, Batch ${batch})` 
    : `Open Race — ${difficulty} Mode (${hiddenFrac*100}% Hidden, Batch ${batch})`;
}

function hasDataForSelection() {
  if (!playgroundData) return false;
  
  const version = els.version.value;
  const batch = els.batch.value; // Keep as string for JSON key access
  const hiddenFrac = els.hiddenFrac.value; // Keep as string for JSON key access
  const difficulty = els.difficulty.value;
  const raceType = version === "hide" ? "Hide_The_Label" : "Open_Race";
  
  try {
    const data = playgroundData[hiddenFrac]?.[difficulty]?.[raceType]?.[batch];
    return data && Object.keys(data).length > 0;
  } catch (e) {
    return false;
  }
}

function generatePlaceholderBarData(selected, seedBase) {
  const labels = selected;
  const colors = getColorPalette(selected.length);
  const rng = mulberry32(seedBase);
  const base = seededRandBetween(rng, 60, 95);
  const data = selected.map(() => Math.max(40, Math.min(100, base + seededRandBetween(rng, -8, 8))));

  return {
    labels,
    datasets: [
      {
        label: "Performance Score (Placeholder Data)",
        data,
        backgroundColor: colors,
        borderColor: colors,
        borderWidth: 1,
      },
    ],
  };
}

function generatePlaceholderLineData(selected, seedBase) {
  const colors = getColorPalette(selected.length);
  const steps = 30;
  const labels = Array.from({ length: steps }, (_, i) => `${i + 1}`);
  const datasets = [];

  selected.forEach((name, idx) => {
    const rng = mulberry32(stringHash(`${seedBase}:${name}`));
    let level = seededRandBetween(rng, 40, 65);
    const drift = seededRandBetween(rng, 0.6, 1.6);
    const noiseAmp = seededRandBetween(rng, 0.8, 2.5);
    const data = [];
    for (let t = 0; t < steps; t++) {
      level += drift + (rng() - 0.5) * noiseAmp;
      data.push(Math.max(0, Math.min(100, level)));
    }
    datasets.push({
      label: name,
      data,
      borderColor: colors[idx],
      backgroundColor: colors[idx],
      fill: false,
      tension: 0.2,
      pointRadius: 1.8,
    });
  });

  return { labels, datasets };
}

function generateBarData(selected) {
  const version = els.version.value;
  const batch = els.batch.value; // Keep as string for JSON key access
  const hiddenFrac = els.hiddenFrac.value; // Keep as string for JSON key access
  const difficulty = els.difficulty.value;
  const selectedDataset = els.dataset.value;
  const raceType = "Hide_The_Label";
  
  const labels = selected;
  const colors = getColorPalette(selected.length);
  
  // Check if we have real data
  if (!playgroundData || !hasDataForSelection()) {
    const seedBase = stringHash(`${version}:${selected.join(',')}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data - No results available for this configuration";
    return generatePlaceholderBarData(selected, seedBase);
  }
  
  try {
    const modeData = playgroundData[hiddenFrac][difficulty][raceType][batch];
    
    // Collect data for selected optimizers
    const data = selected.map(opt => {
      let values = [];
      
      if (selectedDataset === "all") {
        // Average across all datasets
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData && optData.mean_steps !== undefined) {
            values.push(optData.mean_steps);
          }
        }
      } else {
        // Use specific dataset
        const optData = modeData[selectedDataset]?.[opt];
        if (optData && optData.mean_steps !== undefined) {
          values.push(optData.mean_steps);
        }
      }
      
      if (values.length === 0) return 50; // fallback
      
      const avgSteps = values.reduce((a, b) => a + b, 0) / values.length;
      // Invert for visualization: lower steps = better performance = higher bar
      // Normalize to 0-100 scale
      const maxSteps = 200;
      return Math.max(0, Math.min(100, 100 * (1 - avgSteps / maxSteps)));
    });
    
    els.note.textContent = "Interactive visualization of benchmark results";
    
    return {
      labels,
      datasets: [
        {
          label: "Performance Score (lower steps to target = higher score)",
          data,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
        },
      ],
    };
  } catch (e) {
    console.error("Error generating bar data:", e);
    const seedBase = stringHash(`${version}:${selected.join(',')}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data - Error loading results";
    return generatePlaceholderBarData(selected, seedBase);
  }
}

function generateLineData(selected) {
  const version = els.version.value;
  const batch = els.batch.value; // Keep as string for JSON key access
  const hiddenFrac = els.hiddenFrac.value; // Keep as string for JSON key access
  const difficulty = els.difficulty.value;
  const selectedDataset = els.dataset.value;
  const raceType = "Open_Race";
  
  const colors = getColorPalette(selected.length);
  
  // Check if we have real data
  if (!playgroundData || !hasDataForSelection()) {
    const seedBase = stringHash(`${version}:${selected.join(',')}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data - No results available for this configuration";
    return generatePlaceholderLineData(selected, seedBase);
  }
  
  try {
    const modeData = playgroundData[hiddenFrac][difficulty][raceType][batch];
    
    const datasets = [];
    let maxSteps = 0;
    
    // First pass: determine max steps
    selected.forEach(opt => {
      if (selectedDataset === "all") {
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData && optData.steps) {
            maxSteps = Math.max(maxSteps, optData.steps.length);
          }
        }
      } else {
        const optData = modeData[selectedDataset]?.[opt];
        if (optData && optData.steps) {
          maxSteps = Math.max(maxSteps, optData.steps.length);
        }
      }
    });
    
    const labels = Array.from({ length: maxSteps }, (_, i) => `${i}`);
    
    // Second pass: collect data for each optimizer
    selected.forEach((opt, idx) => {
      let optValues = [];
      
      if (selectedDataset === "all") {
        // Collect values from all datasets for this optimizer
        const datasetValues = [];
        for (const dataset in modeData) {
          const optData = modeData[dataset][opt];
          if (optData && optData.values) {
            datasetValues.push(optData.values);
          }
        }
        
        // Average across datasets for each step
        if (datasetValues.length > 0) {
          for (let step = 0; step < maxSteps; step++) {
            const stepVals = datasetValues
              .map(vals => step < vals.length ? vals[step] : null)
              .filter(v => v !== null);
            if (stepVals.length > 0) {
              optValues.push(stepVals.reduce((a, b) => a + b, 0) / stepVals.length);
            }
          }
        }
      } else {
        // Use specific dataset
        const optData = modeData[selectedDataset]?.[opt];
        if (optData && optData.values) {
          optValues = optData.values;
        }
      }
      
      if (optValues.length > 0) {
        datasets.push({
          label: opt,
          data: optValues,
          borderColor: colors[idx],
          backgroundColor: colors[idx],
          fill: false,
          tension: 0.2,
          pointRadius: 1.8,
        });
      }
    });
    
    if (datasets.length === 0) {
      // No data found, use placeholder
      const seedBase = stringHash(`${version}:${selected.join(',')}:${batch}:${hiddenFrac}`);
      els.note.textContent = "Showing placeholder data - No results available for selected optimizers";
      return generatePlaceholderLineData(selected, seedBase);
    }
    
    els.note.textContent = "Interactive visualization of benchmark results";
    
    return { labels, datasets };
  } catch (e) {
    console.error("Error generating line data:", e);
    const seedBase = stringHash(`${version}:${selected.join(',')}:${batch}:${hiddenFrac}`);
    els.note.textContent = "Showing placeholder data - Error loading results";
    return generatePlaceholderLineData(selected, seedBase);
  }
}

function renderChart() {
  const version = els.version.value;
  const selected = getSelectedOptimizers();

  if (selected.length === 0) {
    alert("Please select at least one optimizer.");
    return;
  }

  const ctx = els.canvas.getContext("2d");
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  if (version === "hide") {
    const data = generateBarData(selected);
    chartInstance = new Chart(ctx, {
      type: "bar",
      data,
      options: {
        responsive: true,
        plugins: {
          legend: { display: true },
          title: { display: false },
          tooltip: { enabled: true },
        },
        scales: {
          y: { min: 0, max: 100, title: { display: true, text: "Performance Score" } },
          x: { title: { display: true, text: "Optimizer" } },
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
        plugins: {
          legend: { display: true },
          title: { display: false },
          tooltip: { enabled: true },
        },
        scales: {
          y: { title: { display: true, text: "Best Value So Far" } },
          x: { title: { display: true, text: "Step" } },
        },
        elements: { point: { radius: 2 } },
      },
    });
  }

  updateChartTitle();
}

function resetForm() {
  els.version.value = "hide";
  els.batch.value = "1";
  els.hiddenFrac.value = "0.95";
  els.difficulty.value = "Regular";
  els.dataset.value = "all";
  populateOptimizers();
  updateChartTitle();
}

function onParameterChange() {
  populateOptimizers();
  updateChartTitle();
}

async function loadRealData() {
  try {
    const response = await fetch('playground_data.json');
    if (response.ok) {
      playgroundData = await response.json();
      console.log('Loaded playground data');
      console.log('Available configurations:', Object.keys(playgroundData));
    } else {
      console.warn('Could not load playground_data.json, using placeholder data');
    }
  } catch (e) {
    console.warn('Error loading playground_data.json:', e);
  }
}

async function init() {
  // Load real data first
  await loadRealData();
  
  populateOptimizers();
  updateChartTitle();
  
  // Add event listeners
  els.version.addEventListener("change", onParameterChange);
  els.batch.addEventListener("change", onParameterChange);
  els.hiddenFrac.addEventListener("change", onParameterChange);
  els.difficulty.addEventListener("change", onParameterChange);
  els.dataset.addEventListener("change", updateChartTitle);
  els.generate.addEventListener("click", renderChart);
  els.reset.addEventListener("click", resetForm);

  // Handle URL parameters for direct navigation
  const urlParams = new URLSearchParams(window.location.search);
  const target = urlParams.get('target');
  if (target === 'hide' || target === 'open') {
    els.version.value = target;
    populateOptimizers();
    updateChartTitle();
  }

  window.addEventListener("hashchange", () => handleHashNavigation(true));

  // Render once with defaults
  renderChart();
  // Navigate if hash deep-link present (but don't scroll on page load)
  handleHashNavigation(false);
}

document.addEventListener("DOMContentLoaded", init);
