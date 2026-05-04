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


def strip_up_suffix(name: str) -> str:
    return name.replace("_uP", "")


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


def extract_scaling_data(training_json_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    data = load_json(training_json_path)
    results = data["all_model_results"]

    names = [r["model_name"] for r in results]
    params = np.array([r["total_params"] for r in results], dtype=np.float64)
    losses = np.array([r["final_val_loss"] for r in results], dtype=np.float64)

    return names, params, losses


def plot_up_lr_sweep(
    up_lr_sweep_results_path: Path,
    output_path: Path,
) -> None:
    data = load_json(up_lr_sweep_results_path)

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
    plt.title("μP Tiny Model Learning Rate Sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_up_training_curves(
    up_training_results_path: Path,
    output_path: Path,
) -> None:
    data = load_json(up_training_results_path)
    results = data["all_model_results"]

    plt.figure(figsize=(8, 5))

    for result in results:
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
    plt.title("μP Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_standard_vs_up_scaling(
    standard_training_json_path: Path,
    up_training_json_path: Path,
    output_path: Path,
    fit_output_path: Path,
) -> Dict[str, Any]:
    std_names, std_params, std_losses = extract_scaling_data(standard_training_json_path)
    up_names, up_params, up_losses = extract_scaling_data(up_training_json_path)

    std_a, std_alpha, std_c, std_sse = fit_power_law(std_params, std_losses)
    up_a, up_alpha, up_c, up_sse = fit_power_law(up_params, up_losses)

    x_min = min(std_params.min(), up_params.min())
    x_max = max(std_params.max(), up_params.max())

    x_fit = np.logspace(
        np.log10(x_min),
        np.log10(x_max),
        200,
    )

    std_y_fit = std_a * (x_fit ** (-std_alpha)) + std_c
    up_y_fit = up_a * (x_fit ** (-up_alpha)) + up_c

    plt.figure(figsize=(8, 5))

    plt.scatter(std_params, std_losses, label="Standard observed")
    plt.plot(x_fit, std_y_fit, label=f"Standard fit: alpha={std_alpha:.4f}")

    plt.scatter(up_params, up_losses, label="μP observed")
    plt.plot(x_fit, up_y_fit, label=f"μP fit: alpha={up_alpha:.4f}")

    for name, x, y in zip(std_names, std_params, std_losses):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))

    for name, x, y in zip(up_names, up_params, up_losses):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, -12))

    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Validation Loss after 1 Epoch")
    plt.title("Standard vs μP Scaling Law")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    fit_result = {
        "standard": {
            "a": std_a,
            "alpha": std_alpha,
            "c": std_c,
            "sse": std_sse,
            "model_names": std_names,
            "params": std_params.tolist(),
            "val_losses": std_losses.tolist(),
        },
        "uP": {
            "a": up_a,
            "alpha": up_alpha,
            "c": up_c,
            "sse": up_sse,
            "model_names": up_names,
            "params": up_params.tolist(),
            "val_losses": up_losses.tolist(),
        },
    }

    with open(fit_output_path, "w", encoding="utf-8") as f:
        json.dump(fit_result, f, indent=2)

    return fit_result


def make_extrapolation_prediction(
    fit_result: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    std_sse = fit_result["standard"]["sse"]
    up_sse = fit_result["uP"]["sse"]

    if up_sse <= std_sse:
        selected = "uP"
    else:
        selected = "standard"

    selected_fit = fit_result[selected]

    params = np.array(selected_fit["params"], dtype=np.float64)
    losses = np.array(selected_fit["val_losses"], dtype=np.float64)

    a = selected_fit["a"]
    alpha = selected_fit["alpha"]
    c = selected_fit["c"]

    largest_param = float(params.max())
    extrapolated_param = 10.0 * largest_param

    predicted_loss = float(a * (extrapolated_param ** (-alpha)) + c)

    fitted_losses = a * (params ** (-alpha)) + c
    residuals = losses - fitted_losses

    if len(residuals) > 1:
        residual_std = float(np.std(residuals, ddof=1))
    else:
        residual_std = 0.0

    lower = predicted_loss - 1.96 * residual_std
    upper = predicted_loss + 1.96 * residual_std

    prediction = {
        "selected_fit": selected,
        "reason": "Selected lower SSE power-law fit.",
        "largest_trained_params": largest_param,
        "extrapolated_params": extrapolated_param,
        "predicted_val_loss": predicted_loss,
        "residual_std": residual_std,
        "approx_95_percent_interval": [lower, upper],
        "a": a,
        "alpha": alpha,
        "c": c,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)

    return prediction


def copy_up_model_results_table(
    source_csv: Path,
    target_csv: Path,
) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, target_csv)


def run_up_analysis(
    standard_lr_sweep_results_path: Path,
    standard_training_results_path: Path,
    up_lr_sweep_results_path: Path,
    up_training_results_path: Path,
    up_model_results_csv_path: Path,
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

    up_lr_sweep_plot_path = plots_dir / "uP_lr_sweep.png"
    up_training_curves_plot_path = plots_dir / "uP_training_curves.png"
    comparison_scaling_plot_path = plots_dir / "standard_vs_uP_scaling.png"

    comparison_fit_path = analysis_dir / "standard_vs_uP_scaling_fit.json"
    extrapolation_path = analysis_dir / "extrapolation_prediction.json"

    up_model_results_table_path = tables_dir / "uP_model_results.csv"

    plot_up_lr_sweep(
        up_lr_sweep_results_path=up_lr_sweep_results_path,
        output_path=up_lr_sweep_plot_path,
    )

    plot_up_training_curves(
        up_training_results_path=up_training_results_path,
        output_path=up_training_curves_plot_path,
    )

    fit_result = plot_standard_vs_up_scaling(
        standard_training_json_path=standard_training_results_path,
        up_training_json_path=up_training_results_path,
        output_path=comparison_scaling_plot_path,
        fit_output_path=comparison_fit_path,
    )

    prediction = make_extrapolation_prediction(
        fit_result=fit_result,
        output_path=extrapolation_path,
    )

    copy_up_model_results_table(
        source_csv=up_model_results_csv_path,
        target_csv=up_model_results_table_path,
    )

    print("\n-----------------------------------")
    print("μP Analysis Finished")
    print("-----------------------------------")
    print(f"μP LR sweep plot: {up_lr_sweep_plot_path}")
    print(f"μP training curves: {up_training_curves_plot_path}")
    print(f"Standard vs μP scaling plot: {comparison_scaling_plot_path}")
    print(f"Fit results: {comparison_fit_path}")
    print(f"Extrapolation prediction: {extrapolation_path}")
    print(f"Standard alpha: {fit_result['standard']['alpha']:.6f}")
    print(f"μP alpha: {fit_result['uP']['alpha']:.6f}")
    print(f"Selected extrapolation fit: {prediction['selected_fit']}")
    print(f"Predicted 10x XL val loss: {prediction['predicted_val_loss']:.6f}")

    return {
        "fit_result": fit_result,
        "prediction": prediction,
    }