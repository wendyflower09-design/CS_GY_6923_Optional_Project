import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from trainer import train_one_model


def run_all_models(
    model_configs: List[Dict[str, Any]],
    train_config: Dict[str, Any],
    best_lr: float,
    data_dir: Path,
    meta_path: Path,
    output_dir: Path,
    loss_curves_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    loss_curves_dir = Path(loss_curves_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    loss_curves_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_config in model_configs:
        model_name = model_config["model_name"]

        print("\n-----------------------------------")
        print(f"Train Model: {model_name}")
        print("-----------------------------------")

        loss_curve_path = loss_curves_dir / f"{model_name}_loss.csv"

        result = train_one_model(
            model_config_dict=model_config,
            train_config=train_config,
            learning_rate=best_lr,
            data_dir=data_dir,
            meta_path=meta_path,
            loss_curve_path=loss_curve_path,
        )

        all_results.append(result)

        partial_results_path = output_dir / "training_results_partial.json"
        with open(partial_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    results_json_path = output_dir / "training_results.json"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_lr": best_lr,
                "all_model_results": all_results,
            },
            f,
            indent=2,
        )

    results_csv_path = output_dir / "model_results.csv"

    fieldnames = [
        "model_name",
        "learning_rate",
        "total_params",
        "final_train_loss",
        "final_val_loss",
        "training_time_seconds",
        "max_gpu_memory_allocated_GB",
        "max_gpu_memory_reserved_GB",
        "gpu_total_memory_GB",
        "memory_allocated_percent",
        "memory_reserved_percent",
        "tokens_per_second",
        "batch_size",
        "block_size",
        "tokens_per_step",
        "max_steps",
        "warmup_steps",
        "loss_curve_path",
    ]

    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow({key: result.get(key, None) for key in fieldnames})

    print("\n-----------------------------------")
    print("All Model Training Finished")
    print("-----------------------------------")
    print(f"Results saved to: {results_json_path}")
    print(f"CSV saved to: {results_csv_path}")

    return all_results