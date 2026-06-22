# Robust CNN Classification of Image Sensor Defect Maps Under Small-Data Conditions Through Synthetic Augmentation

## Overview

This repository contains the implementation of a Master's thesis investigating domain-informed synthetic data augmentation for CNN-based classification of CMOS image sensor defect patterns. The work addresses the **cold-start problem** in semiconductor manufacturing: building reliable classifiers when labelled training data is severely limited (9–23 examples per class).

**Key Finding:** Domain-informed synthetic augmentation enables ResNet18 to achieve **0.9458 balanced accuracy** in cross-validation and **0.9760 structured defect recall** on mixed test sets, starting from only 98 real labelled defect maps.

### Thesis Details

- **Title:** Robust CNN Classification of Image Sensor Defect Maps Under Small-Data Conditions Through Synthetic Augmentation
- **Author:** Nuthi Raviteja Pediredla
- **Institution:** Data Science Institute, Hasselt University
- **Academic Year:** 2025–2026
- **Internal Supervisor:** Len Feremans (Hasselt University)
- **External Supervisor:** Wouter Lievens (Gpixel NV, Antwerp, Belgium)
- **Degree:** Master of Statistics and Data Science

---

## Problem Statement
Gpixel NV, a CMOS image sensor manufacturer, faces a critical challenge: new sensor products often have fewer than ten real labelled examples per defect class, yet waiting months to accumulate data is operationally impractical.

This thesis investigates whether **domain-informed synthetic data augmentation** can overcome this constraint by:
1. Leveraging knowledge of sensor hardware periodicity parameters
2. Generating realistic synthetic defect maps procedurally
3. Using these synthetic examples to train robust CNN classifiers

**Dataset:** 1,146 binary 96×192 defect maps across 8 classes (7 structured patterns + 1 non-patterned trash class). Extreme class imbalance: 92.3% trash class.

---

## Methodology

### Synthetic Generator

A domain-grounded generator was developed for each defect class, based on known structural characteristics and hardware periodicity parameters:

| Class | Pattern Type | Parameters |
|-------|--------------|-----------|
| ZEBRA | Wide horizontal bands (120–170 px) | Spacing, density |
| OKAPI | Narrow horizontal bands (55–95 px) | Spacing, density |
| CHEETAH | Dense checkerboard (2 px period) | Coverage, rotation |
| GRID | Periodic dot grid | period_x=12, period_y=16 |
| DIAGONAL | Column-offset repeating pattern | Orientation, offset |
| BLINDS | Horizontal segments (variable width) | Segment count, position |
| MORSE | Morse-code-like dashes in rows | Dash length, spacing |
| NOISE_EMPTY | Random sparse pixels or near-blank | Sparsity |

**Realism:** Synthetic images undergo additional randomisation (Gaussian noise, re-thresholding) to emulate sensor variability.

### Network Architectures

Two architectures were evaluated:

#### BaselineCNN
- 3 convolutional blocks with 3×3 kernels
- Channel progression: 1 → 16 → 32 → 64
- 2 fully connected layers (dropout p=0.3)
- **Parameters:** 2.38M | **Size:** 9.54 MB | **Inference:** 1.41 ms (T4 GPU)
- ✅ Meets embedded hardware deployment target

#### ResNet18 (Single Channel)
- Modified for 1-channel binary input (no ImageNet pretraining)
- Trained from scratch to isolate architectural effects
- **Parameters:** 11.17M | **Size:** 44.78 MB | **Inference:** 5.42 ms (T4 GPU)
- Requires compression for production deployment

---

## Key Results

### Cross-Validation (Real-Only Validation)

| Configuration | Balanced Accuracy | Structured Recall | Accuracy |
|---------------|-------------------|-------------------|----------|
| **ResNet18_1ch_s750** ⭐ | **0.9458 ± 0.0366** | **0.9619** | 0.9396 ± 0.0445 |
| ResNet18_1ch_s1000 | 0.9198 ± 0.0565 | 0.9500 | 0.9192 ± 0.0566 |
| BaselineCNN_s750 | 0.9078 ± 0.0678 | 0.9363 | 0.8996 ± 0.0822 |

### Mixed Test Set (798 images: 98 real + 700 synthetic with independent seed)

| Configuration | Balanced Accuracy | Structured Recall |
|---------------|-------------------|-------------------|
| ResNet18_1ch_s750 | 0.9415 | **0.9760** |
| ResNet18_1ch_s1000 | 0.9431 | 0.9636 |
| BaselineCNN_s750 | 0.9431 | 0.9636 |

### Synthetic Augmentation Gain

The **floor experiment** quantifies the contribution of the synthetic generator:

| Architecture | Real-Only | Best Config | Gain |
|--------------|-----------|-------------|------|
| ResNet18_1ch | 0.2839 | 0.9458 | **+0.662** |
| BaselineCNN | 0.6641 | 0.9078 | **+0.244** |

**Interpretation:** Without synthetic augmentation, ResNet18 achieves near-random performance on an 8-class problem. Synthetic data alone drives the +0.662 point improvement.

### Statistical Comparison

**Wilcoxon signed-rank test** (BaselineCNN_s750 vs ResNet18_1ch_s750):
- W = 0.0
- p = 0.500 (not significant at α = 0.05)
- **Conclusion:** Architecture choice is secondary to data quality for this problem.

### Per-Class Performance on Mixed Test Set

| Class | ResNet18_s750 | BaselineCNN_s750 |
|-------|---------------|------------------|
| NOISE_EMPTY (trash) | 0.700 | 0.800 |
| ZEBRA | 1.000 | 0.955 |
| OKAPI | 0.991 | 0.973 |
| CHEETAH | 0.956 | 0.974 |
| GRID | 0.976 | 0.862 |
| DIAGONAL | 0.918 | 0.991 |
| BLINDS | 0.991 | 0.991 |
| MORSE | 1.000 | 1.000 |

**Note:** GRID and DIAGONAL are the most frequently confused pair due to shared row periodicity; DIAGONAL distinction (one-column offset) is subtle at 96×192 resolution.

---

## Project Structure

```
.
├── README.md                          # This file
├── data/
│   └── defect_maps/                  # Binary 96×192 PNG images, 8 classes
├── src/
│   ├── generator.py                  # Domain-informed synthetic data generator
│   ├── models.py                     # BaselineCNN and ResNet18_1ch architectures
│   ├── train.py                      # Training pipeline with 4-fold cross-validation
│   ├── evaluate.py                   # Metrics: balanced accuracy, macro F1, recall
│   └── visualization.py              # t-SNE, UMAP, Grad-CAM
├── experiments/
│   ├── phase1_baseline/              # Initial architecture search (Colab)
│   ├── phase2_generator/             # Synthetic generator development
│   ├── phase3_transfer_learning/     # ImageNet pretraining experiments (excluded)
│   ├── phase4_robustness/            # Perturbation and interpretability
│   └── phase5_final_pipeline/        # Final architecture comparison (Kaggle)
├── notebooks/
│   ├── exploratory_analysis.ipynb    # Dataset exploration and class visualisation
│   ├── architecture_comparison.ipynb # Final results and comparisons
│   └── feature_space_analysis.ipynb  # t-SNE and UMAP visualisations
├── results/
│   ├── confusion_matrices.png        # Per-architecture and per-testset
│   ├── gradcam_visualisations.png    # Model attention analysis
│   ├── tsne_umap.png                 # Feature space projections
│   └── experiment_log.csv            # Complete 25-experiment log
└── thesis_report.pdf                 # Full Master's thesis (PDF)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, Matplotlib, Scikit-learn, Scikit-image
- Kaggle T4 GPU (or local NVIDIA GPU with CUDA support)

### Setup

```bash
# Clone the repository
git clone https://github.com/RAVI-TJ/CMOS-Defect-Classification-Synthetic-Augmentation.git
cd CMOS-Defect-Classification-Synthetic-Augmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib scikit-learn scikit-image tqdm

# (Optional) Install development tools
pip install jupyter notebook tensorboard
```

---

## Usage

### 1. Generate Synthetic Data

```python
from src.generator import SyntheticGenerator

# Instantiate generator
gen = SyntheticGenerator(seed=42)

# Generate 750 synthetic images per class
synthetic_images = gen.generate_all_classes(n_per_class=750)

# Save to disk
gen.save_batch(synthetic_images, output_dir='data/synthetic/')
```

### 2. Train a Model

```python
from src.train import train_pipeline
from src.models import BaselineCNN, ResNet18_1ch

# Run 4-fold cross-validation
results = train_pipeline(
    architecture='resnet18_1ch',  # or 'baselinecnn'
    synthetic_amount=750,
    epochs=15,
    batch_size=128,
    seed=42
)

print(f"Balanced Accuracy: {results['mean_bal_acc']:.4f} ± {results['std_bal_acc']:.4f}")
```

### 3. Evaluate on Mixed Test Set

```python
from src.evaluate import evaluate_mixed_testset

# Generate independent test synthetic images (different seed)
test_synthetic = gen.generate_all_classes(n_per_class=100, seed=seed+999)

# Evaluate on 98 real + 700 synthetic
metrics = evaluate_mixed_testset(
    model=trained_model,
    real_images=real_test_images,
    synthetic_images=test_synthetic
)

print(f"Structured Recall: {metrics['structured_recall']:.4f}")
```

### 4. Visualise Feature Space

```python
from src.visualization import plot_tsne, plot_umap, plot_gradcam

# t-SNE and UMAP projections
plot_tsne(model, images, labels)
plot_umap(model, images, labels)

# Grad-CAM attention maps
plot_gradcam(model, sample_images, target_layer='features[6]')
```

---

## Experiment Reproduction

All 25 experiments from the thesis are documented in `Appendix A` of the thesis PDF. To reproduce the final Phase 5 results:

```bash
# Run the complete 10-configuration sweep (both architectures × 5 synthetic amounts)
python experiments/phase5_final_pipeline/run_sweep.py \
    --architectures resnet18_1ch baselinecnn \
    --synthetic_amounts 250 500 750 1000 1250 \
    --folds 4 \
    --seed 42
```

---

## Key Findings & Insights

### 1. **Synthetic Generator is the Primary Contribution**
The floor experiment shows that domain knowledge embedded in the generator drives +0.662 balanced accuracy gain for ResNet18. This validates the hypothesis that **carefully designed synthetic data can overcome extreme data scarcity**.

### 2. **Architecture Matters Less Than Data Quality**
Wilcoxon test: p = 0.500. No statistically significant difference between BaselineCNN and ResNet18_1ch when trained on adequate synthetic data. Once sufficient training diversity is provided, both lightweight and deeper architectures perform equally.

### 3. **BaselineCNN is Deployment-Optimal**
- At 9.54 MB, BaselineCNN fits embedded hardware without compression
- 1.41 ms inference time enables real-time defect processing
- Comparable accuracy to ResNet18 (0.9078 vs 0.9458 balanced accuracy)
- **Recommendation:** Deploy BaselineCNN for production systems

### 4. **GRID–DIAGONAL Confusion is Resolution-Limited**
Both classes share row-periodicity; the distinguishing feature (one-column offset across 192 pixels) is genuinely difficult to resolve at this resolution. This is a limitation of the data representation, not the model.

### 5. **Synthetic and Real Images Occupy the Same Feature Space**
t-SNE and UMAP visualisations show real and synthetic images from the same class cluster together. This provides qualitative evidence that the generator produces domain-aligned synthetic patterns.

---

## Industrial Applications

This approach directly supports **Gpixel's new product launches:**

1. When a new sensor arrives with 9–23 labelled defect examples per class
2. Update the synthetic generator with new hardware clock parameters (px, py)
3. Generate 750 synthetic images per class immediately
4. Train BaselineCNN to 0.91+ balanced accuracy within hours
5. Deploy to production quality control pipelines

**No months-long waiting period required.**

---

## Limitations & Future Work

### Limitations
- Dataset contains only 98 real images; cross-validation folds contain ~24 images each
- Single random seed for final architecture comparison (Phase 5)
- GRID–DIAGONAL confusion persists; may require higher resolution or additional features
- Generator relies on expert-defined structural rules; generalisation to novel defect types unclear

### Future Work
1. **Multi-seed robustness:** Repeat full Phase 5 sweep with 5+ independent seeds
2. **Measured sensor statistics:** Calibrate noise and thresholding parameters using actual sensor data (currently empirical)
3. **Higher-resolution analysis:** Investigate 192×384 defect maps to disambiguate GRID–DIAGONAL
4. **Validation on future sensor generations:** Apply framework to next-generation Gpixel sensors
5. **Comparison against internal baseline:** Benchmark against Gpixel's proprietary defect classification system
6. **Ablation studies:** Isolate contribution of each synthetic augmentation mechanism (noise, thresholding, etc.)

---

## Evaluation Metrics Rationale

Four complementary metrics are reported:

| Metric | Primary Use | Rationale |
|--------|------------|-----------|
| **Balanced Accuracy** | Primary metric | Assigns equal weight to all classes; essential for imbalanced data (92.3% trash) |
| **Accuracy (Micro F1)** | Overall reference | Overall classification quality; can be misleading in class-imbalanced settings |
| **Macro F1** | Supplementary | Treats all classes equally; alternative to balanced accuracy |
| **Structured Defect Recall** | Engineering objective | Measures success on the 7 structured classes (industrial goal) |

**Why not Micro F1 alone?** Micro F1 is equivalent to accuracy and is dominated by the majority (trash) class. It does not reflect performance on the structured defect patterns that Gpixel engineers care about.

---

## Reproducibility & Code Quality

- ✅ All 25 experiments documented in complete experiment log (Appendix A)
- ✅ Seed fixed to 42 for deterministic results
- ✅ 4-fold stratified cross-validation ensures representativeness
- ✅ Real-only validation prevents data leakage (synthetic only in training)
- ✅ Mixed test set evaluation with independent random seed (seed+999)
- ✅ All hyperparameters recorded and matched across architectures

---

## References

Full citations are available in the thesis PDF. Key references include:

- He et al. (2016): Residual Learning for Deep Recognition
- Tobin et al. (2017): Domain Randomisation for Sim-to-Real Transfer
- Maaten & Hinton (2008): t-SNE
- McInnes et al. (2018): UMAP
- Brodersen et al. (2010): Balanced Accuracy

---

## License

This work is released under the **MIT License** for research and educational purposes. See `LICENSE` file for details.

**Note:** The dataset contains proprietary defect map images from Gpixel NV and is not included in this repository. Data access should be requested through the thesis supervisors (Len Feremans, Wouter Lievens).

---

## Contact & Attribution

**Thesis Author:** Nuthi Raviteja Pediredla  
**Email:** ravitj.pn@gmail.com 
**Institution:** Data Science Institute, Hasselt University  

**Supervisors:**
- Len Feremans (Hasselt University)
- Wouter Lievels (Gpixel NV, Antwerp, Belgium)

**Computational Resources:** Kaggle Notebooks with dual T4 GPU environment  

---

## Citation

If you use this work in research, please cite:

```bibtex
@mastersthesis{pediredla2026defect,
  author    = {Pediredla, Nuthi Raviteja},
  title     = {Robust CNN Classification of Image Sensor Defect Maps 
              Under Small-Data Conditions Through Synthetic Augmentation},
  school    = {Data Science Institute, Hasselt University},
  year      = {2026},
  advisor   = {Feremans, Len and Lievens, Wouter}
}
```

---

## Acknowledgements

- **Len Feremans** for direct, constructive feedback and guidance throughout the thesis
- **Wouter Lievens & Gpixel NV** for providing the dataset, industrial context, and concrete technical requirements
- **Kaggle** for access to dual T4 GPU infrastructure enabling the full experimental sweep
- The open-source community for PyTorch, scikit-learn, and visualisation tools

---

**Last Updated:** June 2026  

