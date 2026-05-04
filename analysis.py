import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_lr_sweep(
    lr_sweep_results_path: Path,
    output_path: Path,
) -> None:
    data = load_json(lr_sweep_results_path)

    successful = [
        r for r in data["all_results"]
        if r.get("status") == "success"
        and r.get("final_val_loss") is not None
    ]

    lrs = [r["learning_rate"] for r in successful]
    val_losses = [r["final_val_loss"] for r in successful]

    plt.figure(figsize=(7, 5))
    plt.plot(lrs, val_losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Loss")
    plt.title("Tiny Model Learning Rate Sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curves(
    training_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    for result in training_results:
        model_name = result["model_name"]
        loss_curve_path = result.get("loss_curve_path")

        if loss_curve_path is None:
            continue

        steps = []
        losses = []

        with open(loss_curve_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["train_loss"]))

        if len(steps) > 0:
            plt.plot(steps, losses, label=model_name)

    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def fit_power_law(
    params: np.ndarray,
    losses: np.ndarray,
) -> Tuple[float, float, float, float]:
    min_loss = float(losses.min())

    c_min = 0.0
    c_max = min_loss - 1e-5

    if c_max <= c_min:
        c_grid = np.array([0.0])
    else:
        c_grid = np.linspace(c_min, c_max, 1000)

    best_sse = float("inf")
    best_a = None
    best_alpha = None
    best_c = None

    log_params = np.log(params)

    for c in c_grid:
        shifted = losses - c

        if np.any(shifted <= 0):
            continue

        log_losses = np.log(shifted)

        slope, intercept = np.polyfit(log_params, log_losses, deg=1)

        alpha = -slope
        a = math.exp(intercept)

        pred = a * (params ** (-alpha)) + c
        sse = float(np.sum((losses - pred) ** 2))

        if sse < best_sse:
            best_sse = sse
            best_a = a
            best_alpha = alpha
            best_c = c

    return best_a, best_alpha, best_c, best_sse


def plot_scaling_law(
    training_results: List[Dict[str, Any]],
    output_path: Path,
    fit_output_path: Path,
) -> Dict[str, Any]:
    model_names = [r["model_name"] for r in training_results]
    params = np.array([r["total_params"] for r in training_results], dtype=np.float64)
    val_losses = np.array([r["final_val_loss"] for r in training_results], dtype=np.float64)

    a, alpha, c, sse = fit_power_law(params, val_losses)

    x_fit = np.logspace(
        np.log10(params.min()),
        np.log10(params.max()),
        200,
    )

    y_fit = a * (x_fit ** (-alpha)) + c

    plt.figure(figsize=(7, 5))
    plt.scatter(params, val_losses, label="Observed")

    for name, x, y in zip(model_names, params, val_losses):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.plot(x_fit, y_fit, label=f"Fit: alpha={alpha:.4f}")
    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Validation Loss after 1 Epoch")
    plt.title("Transformer Scaling Law")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    fit_result = {
        "a": a,
        "alpha": alpha,
        "c": c,
        "sse": sse,
        "model_names": model_names,
        "params": params.tolist(),
        "val_losses": val_losses.tolist(),
    }

    with open(fit_output_path, "w", encoding="utf-8") as f:
        json.dump(fit_result, f, indent=2)

    return fit_result


def copy_model_results_table(
    source_csv: Path,
    target_csv: Path,
) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, target_csv)


def run_analysis(
    lr_sweep_results_path: Path,
    training_results_path: Path,
    model_results_csv_path: Path,
    analysis_dir: Path,
    plots_dir: Path,
    tables_dir: Path,
) -> Dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    plots_dir = Path(plots_dir)
    tables_dir = Path(tables_dir)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    training_data = load_json(training_results_path)
    training_results = training_data["all_model_results"]

    lr_sweep_plot_path = plots_dir / "lr_sweep.png"
    training_curves_plot_path = plots_dir / "training_curves.png"
    scaling_law_plot_path = plots_dir / "scaling_law.png"
    scaling_fit_path = analysis_dir / "scaling_fit.json"
    model_results_table_path = tables_dir / "model_results.csv"

    plot_lr_sweep(
        lr_sweep_results_path=lr_sweep_results_path,
        output_path=lr_sweep_plot_path,
    )

    plot_training_curves(
        training_results=training_results,
        output_path=training_curves_plot_path,
    )

    fit_result = plot_scaling_law(
        training_results=training_results,
        output_path=scaling_law_plot_path,
        fit_output_path=scaling_fit_path,
    )

    copy_model_results_table(
        source_csv=model_results_csv_path,
        target_csv=model_results_table_path,
    )

    print("\n-----------------------------------")
    print("Analysis Finished")
    print("-----------------------------------")
    print(f"LR sweep plot: {lr_sweep_plot_path}")
    print(f"Training curves plot: {training_curves_plot_path}")
    print(f"Scaling law plot: {scaling_law_plot_path}")
    print(f"Scaling fit: {scaling_fit_path}")
    print(f"Fitted alpha: {fit_result['alpha']:.6f}")

    return fit_result