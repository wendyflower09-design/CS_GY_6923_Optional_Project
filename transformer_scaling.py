from config import (
    DATA_DIR,
    META_PATH,
    LR_SWEEP_DIR,
    ALL_MODELS_DIR,
    LOSS_CURVES_DIR,
    ANALYSIS_DIR,
    PLOTS_DIR,
    TABLES_DIR,
    TINY_MODEL_CONFIG,
    ALL_MODEL_CONFIGS,
    TRAIN_CONFIG,
)

from lr_sweep import run_lr_sweep
from run_all_models import run_all_models
from analysis import run_analysis


def print_step(step_id: int, title: str) -> None:
    print("\n-----------------------------------")
    print(f"Step {step_id}: {title}")
    print("-----------------------------------")


def main():
    print_step(1, "Tiny Model LR Sweep")

    best_lr, _ = run_lr_sweep(
        model_config=TINY_MODEL_CONFIG,
        train_config=TRAIN_CONFIG,
        data_dir=DATA_DIR,
        meta_path=META_PATH,
        output_dir=LR_SWEEP_DIR,
    )

    print_step(2, "Train All Model Sizes")

    all_results = run_all_models(
        model_configs=ALL_MODEL_CONFIGS,
        train_config=TRAIN_CONFIG,
        best_lr=best_lr,
        data_dir=DATA_DIR,
        meta_path=META_PATH,
        output_dir=ALL_MODELS_DIR,
        loss_curves_dir=LOSS_CURVES_DIR,
    )

    print_step(3, "Analyze Results")

    fit_result = run_analysis(
        lr_sweep_results_path=LR_SWEEP_DIR / "lr_sweep_results.json",
        training_results_path=ALL_MODELS_DIR / "training_results.json",
        model_results_csv_path=ALL_MODELS_DIR / "model_results.csv",
        analysis_dir=ANALYSIS_DIR,
        plots_dir=PLOTS_DIR,
        tables_dir=TABLES_DIR,
    )

    print_step(4, "Part 2 Summary")

    print(f"Best learning rate: {best_lr}")
    print(f"Scaling exponent alpha: {fit_result['alpha']:.6f}")

    print("\nModel validation losses:")

    for result in all_results:
        print(
            f"{result['model_name']}: "
            f"params={result['total_params']:,}, "
            f"val_loss={result['final_val_loss']:.4f}, "
            f"time={result['training_time_seconds']} sec, "
            f"memory_reserved={result['max_gpu_memory_reserved_GB']} GB"
        )


if __name__ == "__main__":
    main()