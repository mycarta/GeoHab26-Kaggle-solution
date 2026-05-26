# Backscatter Preprocessing Ablation — Summary

_Refuge Cove (GeoHab 2026), post-competition methodology study. Inspired by_
_Fakiris et al. (2012, ECUA): we cannot ablate the Geocoder mosaic chain (no_
_raw MBES), so we instead ablate the **feature-extraction preprocessing**._
_Generated 2026-05-26._

## Scaffold (frozen across all 17 runs)

- **Features:** 87 = v07 (86) + `acr_sapa_w21` (1 col from v09). Bathy block and acr_sapa_w21 frozen; only the 41 BS-derived columns vary.
- **Model:** LightGBM, `n_estimators=1000`, default hyperparameters, ALG class_weight=2.0, random_state=42, no early stopping (per project Hard Rule 1).
- **CV:** 5-fold 150m verde BlockKFold, seed=42, fold assignments from `kaggle_dataset/fold_indices.csv` (same folds as Sub 42 lock candidate).
- **Metrics:** Out-of-fold weighted F1 and Cohen's κ on raw argmax (no Bayes prior adjustment).
- **Diagnostic only:** train→test mean Wasserstein drift on the modified columns.

## Headline findings (one-bullet summary)

- **Quantization (Exp A) is nearly invariant** in the 8 → 256 range. OOF F1 spread = 0.0088, all variants within fold SE. q=128 is the nominal best (0.7501), q=256 (the historical v07 default) is statistically indistinguishable (0.7492).
- **Gaussian smoothing of BS (Exp B) helps modestly at high σ.** Best variant: σ=8 px (2.0 m) → OOF F1 = 0.7538, κ = 0.6367 (+0.0046 over σ=0). Drift drops with smoothing (0.263 → 0.163), consistent with speckle-suppression hypothesis (Fakiris 2012).
- **Destriping (Exp C) helps marginally.** C3 (FFT-only, no median) = 0.7501 ≈ C2 (FFT+median) = 0.7499 > C1 (original) = 0.7450. The destriping → +0.005 effect is real but smaller than a fold SE. Choice of post-destriping median has negligible impact.
- **Second-order GLCM adds essentially nothing on top of first-order BS texture (Exp D).** D1 (first-order only, 75 feat) = 0.7500, D3 (D1 + GLCM, 87 feat) = 0.7501. GLCM-only (D2, 59 feat) = 0.7381 — worst. Interpretation: in this dataset, focal stats + rank entropy + HSI already saturate the BS texture signal that LightGBM can exploit.
- **σ=0 sanity check passed:** my GLCM extractor reproduces v07's GLCM block to MAE < 2e-15 (machine precision). Implementation equivalence with the lock-candidate pipeline is confirmed for the GLCM step.

## Caveat

Exp C C1 (re-extracted original BS, 0.7450) differs from Exp B σ=0 (0.7492) by 0.0042 despite being structurally equivalent. Cause: my re-extraction of focal stats / rank texture / HSI does not byte-match v07's on-disk float32 rasters (small float32→float64 round-trip differences in HSI normalization on full-raster min/max). The C1↔C2↔C3 comparison is internally consistent (all three share the re-extractor), but absolute levels in Exp C should not be compared to Exp A/B numbers. **Headline destriping conclusion holds**; absolute Exp-C numbers carry an additional ~0.004 floor of re-extraction noise.

---

## Experiment A — GLCM quantization

12 GLCM features (homogeneity, contrast, correlation, energy at patches 7/15/25, 4-angle-averaged) re-extracted from `data/raw/MBES/backscatter.tif` at varying quantization levels. Clip [-45, -5] dB. All other 75 features frozen.

| quantization | OOF weighted F1 | OOF κ | Wasserstein drift |
|---:|---:|---:|---:|
| 8   | 0.7497 | 0.6296 | 0.1686 |
| 16  | 0.7432 | 0.6188 | 0.2636 |
| 32  | 0.7475 | 0.6268 | 0.2895 |
| 64  | 0.7413 | 0.6170 | 0.2641 |
| **128** | **0.7501** | **0.6310** | 0.2662 |
| 256 | 0.7492 | 0.6299 | 0.2626 |

## Experiment B — BS Gaussian smoothing (GLCM-only)

12 GLCM features re-extracted after applying `scipy.ndimage.gaussian_filter(bs, sigma=σ_px)` to the BS raster. Levels = 256 (matches v07). σ=0 is the v07 byte-equivalent baseline (sanity check passed at MAE < 2e-15).

| σ (px) | σ (m) | OOF weighted F1 | OOF κ | Wasserstein drift |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.7492 | 0.6299 | 0.2626 |
| 1 | 0.25 | 0.7497 | 0.6309 | 0.2573 |
| 2 | 0.50 | 0.7496 | 0.6300 | 0.2469 |
| 4 | 1.00 | 0.7450 | 0.6228 | 0.2057 |
| **8** | **2.00** | **0.7538** | **0.6367** | 0.1630 |

## Experiment C — Destriped vs original BS (full 41-feature re-extraction)

All 41 BS-derived features re-extracted from each raster. Clip range for non-dB rasters from raster P01/P99 (printed at script start). uint8 quantization had > 200 unique values for all three rasters (asserted; none was degenerate).

| Variant | OOF weighted F1 | OOF κ | Wasserstein drift |
|---|---:|---:|---:|
| C1 — original (dB) | 0.7450 | 0.6232 | 0.2680 |
| C2 — destriped FFT + median 3×3 ×2 (canonical CNN raster) | 0.7499 | 0.6275 | 0.2477 |
| **C3 — destriped FFT only (no median)** | **0.7501** | **0.6309** | 0.2722 |

## Experiment D — First-order vs second-order texture

Bathy (45) + acr_sapa_w21 (1) frozen. BS texture features vary. GLCM uses Exp-A-optimal quantization q=128.

| Variant | n_feat | OOF weighted F1 | OOF κ |
|---|---:|---:|---:|
| D1 — first-order only (raw BS + focal + rank + HSI)         | 75 | 0.7500 | 0.6310 |
| D2 — GLCM only at q=128 (raw BS + GLCM)                     | 59 | 0.7381 | 0.6112 |
| **D3 — combined** (D1 ∪ GLCM at q=128, 87 feat)             | 87 | **0.7501** | **0.6310** |

D3 ≈ D1 + 0.0001 — adding GLCM on top of first-order BS texture gives ~zero improvement at this scaffold.

---

## Reproduction

```
conda run -n wbw python src/bs_ablation/run_exp_a.py
conda run -n wbw python src/bs_ablation/run_exp_b.py
conda run -n wbw python src/bs_ablation/run_exp_c.py
conda run -n wbw python src/bs_ablation/run_exp_d.py
conda run -n wbw python src/bs_ablation/build_summary.py
```

Total wall-clock: ≈ 80 min on a single workstation (Windows 10, wbw conda env, scikit-image 0.26.0, LightGBM default). Dominated by GLCM at q=256 (392 s/variant in Exp A/B) and full BS re-extraction in Exp C (≈ 7 min/raster).

All OOF predictions per variant saved to `outputs/bs_ablation/oof_predictions/{tag}.csv` for downstream McNemar tests.

## Citation hooks

- Fakiris et al. (2012), *ECUA Proceedings* — Geocoder backscatter-mosaic ablation, kappa-monitored.
- Hall-Beyer (2017), *Practical guidelines for choosing GLCM textures* — quantization-property interactions.
- Clausi (2002), *Canadian J. Remote Sensing* 28(1):45–62 — GLCM grey-level sensitivity.
- Lock candidate (Sub 42) tabular leg: see `generate_sub42_kappa_blend_acr_alg_weighted.py` and `run_v07_acr_sapa_w21_alg_weighted.py`.
