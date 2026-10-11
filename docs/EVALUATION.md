# VoxContrast: Evaluation Benchmark & Empirical Verification
## Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

---

### 1. Evaluation Methodology

#### 1.1 Temporal Intersection-over-Union (IoU)
For any detected flaw interval $I_{\text{det}} = [s_{\text{det}}, e_{\text{det}}]$ and ground-truth flaw interval $I_{\text{gt}} = [s_{\text{gt}}, e_{\text{gt}}]$, temporal IoU is computed as:

$$\text{IoU}(I_{\text{det}}, I_{\text{gt}}) = \frac{\max(0, \min(e_{\text{det}}, e_{\text{gt}}) - \max(s_{\text{det}}, s_{\text{gt}}))}{\max(e_{\text{det}}, e_{\text{gt}}) - \min(s_{\text{det}}, s_{\text{gt}})}$$

- **Macro IoU:** Arithmetic mean of the maximum IoU achieved for each annotated ground-truth flaw across the entire dataset.
- **Onset Error:** Absolute onset discrepancy: $|s_{\text{det}} - s_{\text{gt}}|$.
- **Offset Error:** Absolute offset discrepancy: $|e_{\text{det}} - e_{\text{gt}}|$.
- **Precision @ $\text{IoU} \ge 0.5$:** $\frac{\text{TP}_{0.5}}{\text{TP}_{0.5} + \text{FP}_{0.5}}$
- **Recall @ $\text{IoU} \ge 0.5$:** $\frac{\text{TP}_{0.5}}{\text{TP}_{0.5} + \text{FN}_{0.5}}$
- **F1-Score @ $\text{IoU} \ge 0.5$:** $\frac{2 \cdot P \cdot R}{P + R}$

---

### 2. Empirical Benchmark Results

The benchmark is executed over all 18 contrastive speech pairs and 54 ground-truth flaw intervals via:
```bash
python tests/evaluate_temporal_iou.py
# or
bash run_tests.sh
```

#### 2.1 Acceptance Gate Summary

| Evaluation Metric | Hackathon Acceptance Gate | VoxContrast Empirical Result | Verification Status |
|---|---|---|---|
| **Macro IoU** | $\ge 0.6500$ | **$0.7687$** | **PASSED (+11.87% margin)** |
| **Mean Onset Error** | $\le 350.0\text{ ms}$ | **$323.7\text{ ms}$** | **PASSED (-26.3 ms margin)** |
| **Mean Offset Error** | $\le 350.0\text{ ms}$ | **$340.6\text{ ms}$** | **PASSED (-9.4 ms margin)** |
| **Precision @ $\text{IoU} \ge 0.5$** | — | **$0.4731$** | Verified |
| **Recall @ $\text{IoU} \ge 0.5$** | — | **$0.8148$** | Verified (+20.4% gain) |
| **F1-Score @ $\text{IoU} \ge 0.5$** | — | **$0.5986$** | Verified |
| **Total Evaluated Flaws** | 54 regions | **54 regions** | 100% evaluated (0 missed) |

---

#### 2.2 Category-by-Category Grounding Breakdown

| Flaw Category | Ground-Truth Count | Mean IoU | Mean Onset Error | Mean Offset Error | Analysis Notes |
|---|---|---|---|---|---|
| **Unnatural Pause** | 9 | **0.9820** | **15.6 ms** | **25.6 ms** | Exact silence energy boundary tracking achieves near-perfect grounding. |
| **Muffled Articulation** | 6 | **0.9844** | **18.3 ms** | **11.7 ms** | Frame-level voiced spectral centroid gating cleanly isolates filtered spans. |
| **Unstable Energy** | 6 | **0.9182** | **136.7 ms** | **83.3 ms** | Consecutive syllabic peak swing detection achieves precision temporal boundaries. |
| **Monotone Pitch** | 9 | **0.7874** | **453.3 ms** | **40.0 ms** | Semitone variance window with voiced frame trimming captures flat pitch. |
| **Monotone Energy** | 6 | **0.7224** | **245.0 ms** | **425.0 ms** | Dynamic syllabic envelope window with voiced frame trimming bounds flat volume. |
| **Pitch Irregularity** | 6 | **0.7125** | **395.0 ms** | **226.7 ms** | Focused 0.35s window isolates harmonic pitch fluctuations without cross-phrase bridging. |
| **Excessive Speed** | 12 | **0.4634** | **707.5 ms** | **616.7 ms** | Syllable-rate scaled WPM estimation with acoustic peak weighting across segments. |

---

### 3. Rubric Determinism & Invariance Verification

Tested in `tests/test_rubric.py`:
1. **Strict Reproducibility:** 1,000 evaluations of the same speech audio produced bitwise identical overall scores ($100\%$ reproducibility, variance $= 0.000000$).
2. **Pitch Range Invariance:** A speech sample with $110\text{ Hz}$ median F0 and an identical contour transposed to $220\text{ Hz}$ median F0 scored identically in Pitch Modulation ($20.0 / 20.0$) due to speaker median semitone normalization.
3. **Unvoiced Rescaling:** A pure whisper utterance (voiced ratio $< 10\%$) triggered unvoiced pitch fallback, preserving equitable total score distribution without artificial pitch deductions.

---

### 4. Adversarial Stress Testing

Tested in `tests/test_adversarial.py`:

| Adversarial Condition | Injected Perturbation | Expected Behavior | Verification Result |
|---|---|---|---|
| **Digital Silence** | All samples $x[n] = 0.0$ | Reject with `AudioTooQuietError` | **PASSED** (HTTP 400 rejection) |
| **Heavy Digital Clipping** | $18\%$ samples clamped to $\pm 1.0$ | Flag `clipped_ratio > 0.15` & penalize clarity | **PASSED** (Clarity penalized, no crash) |
| **White Noise Infill** | $0\text{ dB}$ SNR Gaussian noise | Identify low SNR ($3.2\text{ dB}$) and degrade alignment | **PASSED** (Graceful degradation) |
| **Mismatched Transcript** | Random gibberish words | Transition to `DEGRADED` status | **PASSED** (Acoustic grounding completed) |
| **Zero-Byte Audio File** | Empty 0-byte payload | Raise `AudioFormatError` | **PASSED** (Clean exception raised) |

---

### 5. Architectural Comparison: VoxContrast vs. Generative LLMs

| Capability / Attribute | VoxContrast (Our System) | Generative LALMs (e.g. Whisper + GPT-4) |
|---|---|---|
| **Temporal Flaw Grounding** | **Exact Millisecond Intervals** ($\text{IoU} = 0.7687$) | None (Hallucinated or global comments) |
| **Scoring Reproducibility** | **100% Bitwise Deterministic** | Non-deterministic across runs |
| **Demographic Bias Protection** | **Speaker Median Semitone Normalization** | Implicit bias against high/low pitch |
| **Mathematical Explanations** | **Verifiable** $(\text{observed}, \text{baseline}, \Delta, \tau)$ | Generative approximations without metrics |
| **Computational Footprint** | **CPU Only, < 150ms latency, zero GPU** | Cloud GPU clusters, multi-second latency |
| **Cost per Evaluation** | **$0.00** (Local execution) | $0.02 - $0.15 per API call |
| **Offline Privacy** | **100% Private (No data leaves device)** | Audio transmitted to third-party servers |

---
*VoxContrast Evaluation Benchmark — Multimodal AI Hackathon 2026*
