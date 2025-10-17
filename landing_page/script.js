"use strict";

// Known optimizers from the project (baseline set)
const AVAILABLE_OPTIMIZERS = [
  "RANDOM",
  "BO_GP_EI",
  "SMART_BO",
  "SBO_GP_PV",
  "SBO_ANN_PV",
  "SBO_POLY_PV",
  "SBO_GP_EI_TRUNCDE",
  "DE_DIRECT",
  "FULL_FACTORIAL",
  "FRACTIONAL_FACTORIAL",
  "PLACKETT_BURMAN",
  "CENTRAL_COMPOSITE",
  "BOX_BEHNKEN",
  "LATIN_HYPERCUBE",
  "D_OPTIMAL",
];

let chartInstance = null;

const els = {
  version: document.getElementById("versionSelect"),
  batch: document.getElementById("batchInput"),
  initialCfg: document.getElementById("initialConfigSelect"),
  optimizers: document.getElementById("optimizersContainer"),
  generate: document.getElementById("generateBtn"),
  reset: document.getElementById("resetBtn"),
  canvas: document.getElementById("resultsChart"),
  title: document.getElementById("chartTitle"),
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

function populateOptimizers() {
  els.optimizers.innerHTML = "";
  AVAILABLE_OPTIMIZERS.forEach((name, idx) => {
    const id = `opt_${idx}`;
    const wrapper = document.createElement("label");
    wrapper.className = "opt-item";
    wrapper.setAttribute("for", id);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = id;
    checkbox.value = name;
    if (idx < 5) checkbox.checked = true; // default some selections

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
  els.title.textContent = isHide ? "Hide the Label — Results" : "Open Race — Results";
}

function generateBarData(selected, seedBase, initialCfg, batchNum) {
  const rng = mulberry32(seedBase);
  // base between 60 and 95, slight boosts for 100% and larger batch
  const base = seededRandBetween(rng, 60, 95);
  const boost = (initialCfg >= 1.0 ? 3 : 0) + Math.min(4, Math.log2(Math.max(1, batchNum)));
  const labels = selected;
  const colors = getColorPalette(selected.length);
  const data = selected.map(() => Math.max(40, Math.min(100, base + seededRandBetween(rng, -8, 8) + boost)));

  return {
    labels,
    datasets: [
      {
        label: "Final score (placeholder)",
        data,
        backgroundColor: colors,
        borderColor: colors,
        borderWidth: 1,
      },
    ],
  };
}

function generateLineData(selected, seedBase, initialCfg, batchNum) {
  const steps = Math.min(50, Math.max(12, 10 + batchNum * 5));
  const labels = Array.from({ length: steps }, (_, i) => `${i + 1}`);
  const colors = getColorPalette(selected.length);
  const datasets = [];

  selected.forEach((name, idx) => {
    const rng = mulberry32(stringHash(`${seedBase}:${name}`));
    let level = seededRandBetween(rng, 40, 65);
    const drift = seededRandBetween(rng, 0.6, 1.6) + (initialCfg >= 1.0 ? 0.2 : 0);
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

function renderChart() {
  const version = els.version.value; // "hide" | "open"
  const batchNum = Math.max(1, parseInt(els.batch.value || "1", 10));
  const initialCfg = parseFloat(els.initialCfg.value || "0.95");
  const selected = getSelectedOptimizers();

  if (selected.length === 0) {
    alert("Please select at least one optimizer.");
    return;
  }

  const seedBase = stringHash(`${version}:${selected.join(',')}:${batchNum}:${initialCfg}`);

  const ctx = els.canvas.getContext("2d");
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  if (version === "hide") {
    const data = generateBarData(selected, seedBase, initialCfg, batchNum);
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
          y: { min: 0, max: 100, title: { display: true, text: "Score (placeholder)" } },
          x: { title: { display: true, text: "Optimizer" } },
        },
      },
    });
  } else {
    const data = generateLineData(selected, seedBase, initialCfg, batchNum);
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
          y: { min: 0, max: 100, title: { display: true, text: "Score (placeholder)" } },
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
  els.initialCfg.value = "0.95";
  Array.from(els.optimizers.querySelectorAll('input[type="checkbox"]')).forEach((c, idx) => {
    c.checked = idx < 5;
  });
  updateChartTitle();
}

function init() {
  populateOptimizers();
  updateChartTitle();
  els.version.addEventListener("change", updateChartTitle);
  els.generate.addEventListener("click", renderChart);
  els.reset.addEventListener("click", resetForm);

  // Handle URL parameters for direct navigation
  const urlParams = new URLSearchParams(window.location.search);
  const target = urlParams.get('target');
  if (target === 'hide' || target === 'open') {
    els.version.value = target;
    updateChartTitle();
  }

  window.addEventListener("hashchange", () => handleHashNavigation(true));

  // Render once with defaults
  renderChart();
  // Navigate if hash deep-link present (but don't scroll on page load)
  handleHashNavigation(false);
}

document.addEventListener("DOMContentLoaded", init);




