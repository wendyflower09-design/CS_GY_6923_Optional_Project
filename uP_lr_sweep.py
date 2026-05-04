import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from uP_trainer import train_one_up_model


UP_LR_CANDIDATES = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]


def run_up_lr_sweep(
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    data_dir: Path,
    meta_path: Path,
    output_dir: Path,
) -> Tuple[float, List[Dict[str, Any]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for lr in UP_LR_CANDIDATES:
        print("\n-----------------------------------")
        print(f"μP LR Sweep: learning_rate = {lr}")
        print("-----------------------------------")

        try:
            result = train_one_up_model(
                model_config_dict=model_config,
                train_config=train_config,
                learning_rate=lr,
                data_dir=data_dir,
                meta_path=meta_path,
                loss_curve_path=None,
            )

            result["status"] = "success"

        except Exception as e:
            print(f"\nμP LR {lr} failed.")
            print(f"Reason: {str(e)}")

            result = {
                "model_name": model_config["model_name"],
                "parameterization": "uP",
                "learning_rate": lr,
                "status": "failed",
                "error": str(e),
                "final_train_loss": None,
                "final_val_loss": None,
                "training_time_seconds": None,
                "max_gpu_memory_allocated_GB": None,
                "max_gpu_memory_reserved_GB": None,
                "gpu_total_memory_GB": None,
                "memory_allocated_percent": None,
                "memory_reserved_percent": None,
                "tokens_per_second": None,
                "total_params": None,
                "batch_size": train_config["batch_size"],
                "block_size": model_config["block_size"],
                "tokens_per_step": train_config["batch_size"] * model_config["block_size"],
                "max_steps": None,
                "warmup_steps": None,
                "loss_curve_path": None,
            }

        all_results.append(result)

    successful_results = [
        result for result in all_results
        if result["status"] == "success"
        and result["final_val_loss"] is not None
    ]

    if len(successful_results) == 0:
        raise RuntimeError("All μP LR sweep runs failed. No best learning rate found.")

    best_result = min(successful_results, key=lambda x: x["final_val_loss"])
    best_lr = best_result["learning_rate"]

    results_json_path = output_dir / "uP_lr_sweep_results.json"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_lr": best_lr,
                "best_result": best_result,
                "all_results": all_results,
            },
            f,
            indent=2,
        )

    summary_csv_path = output_dir / "uP_lr_sweep_summary.csv"

    fieldnames = [
        "status",
        "learning_rate",
        "final_train_loss",
        "final_val_loss",
        "training_time_seconds",
        "max_gpu_memory_allocated_GB",
        "max_gpu_memory_reserved_GB",
        "gpu_total_memory_GB",
        "memory_allocated_percent",
        "memory_reserved_percent",
        "tokens_per_second",
        "total_params",
        "batch_size",
        "block_size",
        "tokens_per_step",
        "max_steps",
        "warmup_steps",
        "error",
    ]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(
                {key: result.get(key, None) for key in fieldnames}
            )

    print("\n-----------------------------------")
    print("μP LR Sweep Finished")
    print("-----------------------------------")
    print(f"Best μP LR: {best_lr}")
    print(f"Best μP validation loss: {best_result['final_val_loss']:.4f}")
    print(f"Results saved to: {results_json_path}")
    print(f"Summary saved to: {summary_csv_path}")

    return best_lr, all_results