from uP_config import (
    DATA_DIR,
    META_PATH,
    STANDARD_PART2_DIR,
    UP_LR_SWEEP_DIR,
    UP_ALL_MODELS_DIR,
    UP_LOSS_CURVES_DIR,
    UP_ANALYSIS_DIR,
    UP_PLOTS_DIR,
    UP_TABLES_DIR,
    UP_TINY_MODEL_CONFIG,
    UP_ALL_MODEL_CONFIGS,
    UP_TRAIN_CONFIG,
)

from uP_lr_sweep import run_up_lr_sweep
from uP_run_all_models import run_all_up_models
from uP_analysis import run_up_analysis


def print_step(step_id: int, title: str) -> None:
    print("\n-----------------------------------")
    print(f"Part 3 Step {step_id}: {title}")
    print("-----------------------------------")


def main():
    print_step(1, "μP Tiny Model LR Sweep")

    best_up_lr, _ = run_up_lr_sweep(
        model_config=UP_TINY_MODEL_CONFIG,
        train_config=UP_TRAIN_CONFIG,
        data_dir=DATA_DIR,
        meta_path=META_PATH,
        output_dir=UP_LR_SWEEP_DIR,
    )

    print_step(2, "Train All μP Model Sizes")

    up_results = run_all_up_models(
        model_configs=UP_ALL_MODEL_CONFIGS,
        train_config=UP_TRAIN_CONFIG,
        best_lr=best_up_lr,
        data_dir=DATA_DIR,
        meta_path=META_PATH,
        output_dir=UP_ALL_MODELS_DIR,
        loss_curves_dir=UP_LOSS_CURVES_DIR,
    )

    print_step(3, "Analyze Standard vs μP Scaling")

    analysis_result = run_up_analysis(
        standard_lr_sweep_results_path=STANDARD_PART2_DIR / "lr_sweep" / "lr_sweep_results.json",
        standard_training_results_path=STANDARD_PART2_DIR / "all_models" / "training_results.json",
        up_lr_sweep_results_path=UP_LR_SWEEP_DIR / "uP_lr_sweep_results.json",
        up_training_results_path=UP_ALL_MODELS_DIR / "uP_training_results.json",
        up_model_results_csv_path=UP_ALL_MODELS_DIR / "uP_model_results.csv",
        analysis_dir=UP_ANALYSIS_DIR,
        plots_dir=UP_PLOTS_DIR,
        tables_dir=UP_TABLES_DIR,
    )

    print_step(4, "Part 3 Summary")

    fit_result = analysis_result["fit_result"]
    prediction = analysis_result["prediction"]

    print(f"Best μP learning rate: {best_up_lr}")
    print(f"Standard alpha: {fit_result['standard']['alpha']:.6f}")
    print(f"μP alpha: {fit_result['uP']['alpha']:.6f}")
    print(f"Selected extrapolation fit: {prediction['selected_fit']}")
    print(f"Predicted 10x XL validation loss: {prediction['predicted_val_loss']:.6f}")

    print("\nμP model validation losses:")

    for result in up_results:
        print(
            f"{result['model_name']}: "
            f"params={result['total_params']:,}, "
            f"val_loss={result['final_val_loss']:.4f}, "
            f"time={result['training_time_seconds']} sec, "
            f"memory_reserved={result['max_gpu_memory_reserved_GB']} GB"
        )


if __name__ == "__main__":
    main()