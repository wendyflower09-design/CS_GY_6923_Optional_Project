# CS-GY 6923: Optional Project

---

# Environment Setup

```bash
conda create -n svg_project python=3.11
conda activate svg_project
```

```bash
conda install -c conda-forge cairosvg cairo pango gdk-pixbuf
conda install numpy=1.26
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

```bash
pip install -r requirements.txt
```

---

# Full Pipeline

Runs all Part1, Part 2, Part 3, and Part 4: Data preprocessing, standard transformer scaling, μP scaling comparison, and multi-epoch best-model training.

```powershell
.\run_all.ps1
```

This can take 15-20 hours and requires at least 20 GB GPU memory with cuda available.

---

# Part 1: Data Collection and Preprocessing

Downloads the SVG datasets, normalizes the datasets, trains the tokenizer, and encodes and writes tokenized train / validation / test files.

```bash
python data_preprocessing.py
```

```text
data_preprocessing.py   # main preprocessing pipeline
data_download.py        # HuggingFace data loading
normalization.py        # SVG cleaning and validation
tokenization.py         # BPE tokenizer and encoding
stats.py                # dataset statistics
```

Output:

```text
processed_svg_data/
```

---

# Part 2: Standard Transformer Scaling

Runs LR sweep on the tiny model, trains all standard model sizes, and fits a scaling law.

```bash
python transformer_scaling.py
```

```text
transformer_scaling.py  # Part 2 entry point
config.py               # standard model configs
models.py               # standard Transformer LM
trainer.py              # standard training loop
lr_sweep.py             # tiny LR sweep
run_all_models.py       # train all model sizes
analysis.py             # plots and scaling fit
```

Output:

```text
part2_outputs/
```

---

# Part 3: μP Scaling

Runs μP LR sweep, trains all μP model sizes, and compares μP scaling with Part 2.

```bash
python uP_scaling.py
```

```text
uP_scaling.py           # Part 3 entry point
uP_config.py            # μP model configs
uP_model.py             # μP Transformer LM
uP_trainer.py           # μP training loop
uP_lr_sweep.py          # μP tiny LR sweep
uP_run_all_models.py    # train μP model sizes
uP_analysis.py          # standard vs μP analysis
```

Output:

```text
part3_uP_outputs/
```

---

# Part 4: Best Model Training and Generation

Trains the selected XL μP model for multiple epochs using the best Part 3 learning rate.

```bash
python best_model_train.py
```

Generate and evaluate SVGs:

```bash
python generate_svg.py
python evaluate_svg.py
python make_sample_grid.py
```

```text
best_model_train.py     # multi-epoch XL μP training
generate_svg.py         # SVG sampling
evaluate_svg.py         # render and test metrics
make_sample_grid.py     # sample figures
```

Output:

```text
part4_outputs/
```

---

# Debug Utilities

```text
debug_tiny_generation.py       # tiny overfit/generation test
debug_generation_sweep.py      # generation parameter sweep
debug_tokenizer_roundtrip.py   # tokenizer roundtrip check
tokenizer_testing.py           # tokenizer experiment
test_xl_memory.py              # XL memory check
```
