#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Hide-the-Label mean steps per optimizer from a results JSON.

Supports two structures:
- optimizer_stats: { OPT: { mean_steps, std_steps, success_rate, ... }, ... }
- tournament/competitions: competitions[*].optimizer_results[OPT].steps_to_target

Usage:
  Set the variables below (json_file, out_path, plot_title) and run the script.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def extract_from_optimizer_stats(data: Dict) -> Tuple[List[str], List[float], List[float]]:
    stats = data.get("optimizer_stats", {}) or {}
    if not isinstance(stats, dict) or not stats:
        return [], [], []
    optimizers: List[str] = []
    means: List[float] = []
    ses: List[float] = []
    for name, st in stats.items():
        if not isinstance(st, dict):
            continue
        mean_val = st.get("mean_steps")
        if mean_val is None:
            continue
        std_val = st.get("std_steps")
        n = st.get("n_competitions") or st.get("count") or 0
        se = 0.0
        if std_val is not None and isinstance(n, (int, float)) and n and n > 0:
            try:
                se = float(std_val) / float(n) ** 0.5
            except Exception:
                se = 0.0
        optimizers.append(name)
        means.append(float(mean_val))
        ses.append(float(se))
    return optimizers, means, ses


def _collect_steps_by_optimizer(data: Dict) -> Dict[str, List[float]]:
    # Try direct results layout first
    results_node = data.get("results", {}) if isinstance(data.get("results"), dict) else {}
    tournaments = results_node.get("all_tournament_results") or results_node.get("tournament_results", {}).get("competitions") or []
    items: List[Dict] = []
    if isinstance(tournaments, list):
        for t in tournaments:
            comps = t.get("competitions", []) if isinstance(t, dict) else []
            items.extend(comps)
    elif isinstance(tournaments, dict):
        items = tournaments.get("competitions", [])

    # fallback: some files store competitions at root
    if not items:
        items = data.get("competitions", [])

    # handle root-level list of tournaments: data["tournament_results"] -> [{ competitions: [...] }, ...]
    if not items:
        tr = data.get("tournament_results")
        if isinstance(tr, list):
            for t in tr:
                comps = t.get("competitions", []) if isinstance(t, dict) else []
                items.extend(comps)

    # handle aggregated compiled format: top-level all_tournament_results: [ { competitions: [...] }, ... ]
    if not items:
        atr = data.get("all_tournament_results")
        if isinstance(atr, list):
            for t in atr:
                comps = t.get("competitions", []) if isinstance(t, dict) else []
                items.extend(comps)

    steps_by_opt: Dict[str, List[float]] = {}
    for comp in items:
        opt_res = comp.get("optimizer_results", {}) if isinstance(comp, dict) else {}
        if not isinstance(opt_res, dict):
            continue
        for name, res in opt_res.items():
            if not isinstance(res, dict):
                continue
            steps = res.get("steps_to_target")
            if steps is None:
                continue
            steps_by_opt.setdefault(name, []).append(float(steps))
    return steps_by_opt


def extract_from_competitions(data: Dict) -> Tuple[List[str], List[float], List[float]]:
    steps_by_opt = _collect_steps_by_optimizer(data)
    if not steps_by_opt:
        return [], [], []
    optimizers = list(steps_by_opt.keys())
    means = [float(np.mean(steps_by_opt[o])) for o in optimizers]
    stds = [float(np.std(steps_by_opt[o], ddof=1)) if len(steps_by_opt[o]) > 1 else 0.0 for o in optimizers]
    ns = [len(steps_by_opt[o]) for o in optimizers]
    ses = [s / (n ** 0.5) if n > 0 else 0.0 for s, n in zip(stds, ns)]
    return optimizers, means, ses


def plot_bar(optimizers: List[str], means: List[float], ses: List[float], title: str, out_path: str) -> None:
    # sort by mean ascending
    order = np.argsort(means)
    opts = [optimizers[i] for i in order]
    vals = [means[i] for i in order]
    errs = [ses[i] for i in order]

    plt.figure(figsize=(12, 7))
    # Apply viridis gradient (purple→blue→green→yellow) explicitly per bar
    try:
        cmap = plt.cm.get_cmap("viridis")
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis")
    colors = [cmap(t) for t in np.linspace(0.0, 1.0, max(3, len(opts)))][:len(opts)]
    bars = plt.bar(opts, vals, yerr=errs, capsize=4, alpha=0.85)
    for bar, col in zip(bars, colors):
        bar.set_facecolor(col)
    plt.ylabel("Mean steps to target (±1 SE)")
    plt.xlabel("Optimizer")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    # value labels
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals) * 0.02,
                 f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # User-set variables (no CLI flags)
    json_file = "/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/results_currentlyfor_hide_and_open/compiled_hide_label_MOBO_dataset_rat_myocyte_20250909_000000.json"  # <- set this
    out_path = "visualizations/hide_label_mean_steps.png"  # <- set this
    plot_title = "Hide-the-Label: Mean Steps per Optimizer"  # <- set this

    # If someone still passes flags, allow them to override variables silently
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--results")
        parser.add_argument("--out")
        parser.add_argument("--title")
        known, _ = parser.parse_known_args()
        if known.results:
            json_file = known.results
        if known.out:
            out_path = known.out
        if known.title:
            plot_title = known.title
    except Exception:
        pass

    with open(json_file, "r") as f:
        data = json.load(f)

    optimizers, means, ses = extract_from_optimizer_stats(data)
    if not optimizers:
        optimizers, means, ses = extract_from_competitions(data)

    # Fallback: compiled wrapper { files: [ { data: {...} }, ... ] }
    if not optimizers and isinstance(data.get("files"), list):
        merged: Dict[str, List[float]] = {}
        for item in data["files"]:
            if not isinstance(item, dict):
                continue
            d = item.get("data")
            if not isinstance(d, dict):
                continue
            local = _collect_steps_by_optimizer(d)
            for opt, steps in local.items():
                merged.setdefault(opt, []).extend(steps)
        if merged:
            optimizers = list(merged.keys())
            means = [float(np.mean(merged[o])) for o in optimizers]
            stds = [float(np.std(merged[o], ddof=1)) if len(merged[o]) > 1 else 0.0 for o in optimizers]
            ns = [len(merged[o]) for o in optimizers]
            ses = [s / (n ** 0.5) if n > 0 else 0.0 for s, n in zip(stds, ns)]

    if not optimizers:
        raise SystemExit("Could not find optimizer step data in the provided JSON")

    plot_bar(optimizers, means, ses, plot_title, out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


