# VoxContrast: Phase 4 Error Analysis & Generalization Diagnosis
## Contrastive Speech Analytics & Temporal Flaw Grounding (Track C)

---

### Executive Summary

An exhaustive empirical audit of VoxContrast across the frozen benchmark (`data/manifest.json`), the initial TTS-assisted expansion split (`data/tts_expanded/manifest.json`), and clean baseline recordings revealed critical insights into why performance on synthetic glottal audio differed markedly from speech synthesized via system TTS (macOS `say`) and natural continuous speech.

While the core temporal grounding mechanism achieved high Macro IoU ($\text{IoU} = 0.8689$) on the frozen benchmark, precision dropped significantly on TTS speech ($\text{Prec} = 0.1000$ on validation, $\text{Prec} = 0.0930$ on held-out test), with an elevated clean baseline false-positive rate ($40.06\text{ FP/min}$).

This diagnostic investigation proved that reduced precision was **not** caused by genuine inability to ground flaws (Recall remained $80.0\% - 100.0\%$), but by **four distinct acoustic and algorithmic root causes**:
1. Spectral centroid threshold mismatch between noise-padded synthetic glottal waveforms and clean harmonic speech.
2. Non-causal, high-variance local WPM estimation over micro-windows ($0.5\text{s}$) triggering on isolated function words.
3. Word-level alignment rate inflation on short monosyllabic words ($100-150\text{ms}$) assuming uniform 400ms word baselines.
4. Pause-decay transient misclassification as "slow speech" during phoneme transitions into silence.

---

### 1. Recording-by-Recording Diagnostic Audit (TTS Expansion Corpus)

Every recording in `data/tts_expanded/` was audited for duration, active speech content, ground-truth flaws, predictions, and IoU match status at IoU $\ge 0.50$.

#### 1.1 Flawed Contrastive Utterances (`data/tts_expanded/audio/flawed/`)

| Recording ID | Expected Category | Total Dur | Active Dur | GT Flaws | Dets | TP | FP | FN | Primary FP Sources |
|---|---|---|---|---|---|---|---|---|---|
| `speech_07_roosevelt_excessive_speed_severe.wav` | `excessive_speed` | 11.64s | 11.20s | 1 (CONF) | 7 | 1 | 6 | 0 | 1 speed FP (phrase 1), 3 muffled FP, 1 pause FP |
| `speech_07_roosevelt_unnatural_pause_severe.wav` | `unnatural_pause` | 14.78s | 11.95s | 1 (CONF) | 8 | 1 | 7 | 0 | 3 speed FP, 3 muffled FP, 1 monotone energy FP |
| `speech_07_roosevelt_monotone_pitch_severe.wav` | `monotone_pitch` | 12.53s | 11.90s | 1 (CONF) | 8 | 1 | 7 | 0 | 3 speed FP, 3 muffled FP, 1 monotone energy FP |
| `speech_08_curie_slow_delivery_moderate.wav` | `slow_delivery` | 14.03s | 12.60s | 1 (UNCERT) | 7 | 0 | 7 | 0 | Skipped (UNCERTAIN status); 3 speed FP, 3 muffled FP |
| `speech_08_curie_unstable_energy_severe.wav` | `unstable_energy` | 13.38s | 12.25s | 1 (CONF) | 9 | 0 | 9 | 1 | Swing IoU = 0.399 (FN); 4 speed FP, 4 muffled FP |
| `speech_08_curie_monotone_energy_moderate.wav` | `monotone_energy` | 13.38s | 12.25s | 1 (CONF) | 8 | 1 | 7 | 0 | 3 speed FP, 3 muffled FP, 1 pause FP |
| `speech_09_mandela_excessive_speed_severe.wav` | `excessive_speed` | 11.48s | 11.15s | 1 (CONF) | 8 | 1 | 7 | 0 | 5 speed FP, 1 muffled FP, 1 monotone pitch FP |
| `speech_09_mandela_unnatural_pause_severe.wav` | `unnatural_pause` | 14.55s | 11.75s | 1 (CONF) | 14 | 1 | 13 | 0 | 7 speed FP, 3 muffled FP, 2 monotone pitch FP |
| `speech_09_mandela_unstable_energy_moderate.wav` | `unstable_energy` | 12.30s | 11.90s | 1 (CONF) | 12 | 1 | 11 | 0 | 6 speed FP, 3 muffled FP, 1 monotone pitch FP |
| `speech_10_turing_monotone_pitch_severe.wav` | `monotone_pitch` | 12.67s | 11.90s | 1 (CONF) | 9 | 1 | 8 | 0 | 3 speed FP, 3 muffled FP, 1 pitch FP, 1 pause FP |
| `speech_10_turing_slow_delivery_moderate.wav` | `slow_delivery` | 13.32s | 12.45s | 1 (UNCERT) | 8 | 0 | 8 | 0 | Skipped (UNCERTAIN status); 3 speed FP, 4 muffled FP |
| `speech_10_turing_monotone_energy_moderate.wav` | `monotone_energy` | 12.67s | 11.90s | 1 (UNCERT) | 8 | 0 | 8 | 0 | Skipped (UNCERTAIN status); 3 speed FP, 4 muffled FP |

#### 1.2 Detection Count by Category Across All Flawed TTS Recordings

- `muffled_articulation`: **41 detections** (3.4 per recording) — 100% false positives
- `excessive_speed`: **38 detections** (3.2 per recording) — 36 false positives, 2 true positives
- `monotone_energy`: **13 detections** — 11 false positives, 2 true positives
- `monotone_pitch`: **5 detections** — 3 false positives, 2 true positives
- `pitch_irregularity`: **5 detections** — 5 false positives
- `unnatural_pause`: **2 detections** — 0 false positives, 2 true positives
- `unstable_energy`: **2 detections** — 1 false positive, 1 true positive

---

### 2. Clean Baseline Speech Audit (Zero Flaw Reference)

Evaluating pristine baseline audio files where no defect was injected:

| Baseline Recording | Duration | Active Speech | Total Dets (All FP) | FP/min (Total) | FP/min (Active) | Category Breakdown |
|---|---|---|---|---|---|---|
| `speech_07_roosevelt_baseline.wav` | 11.73s | 11.40s | 9 | 46.0 | 47.4 | 4 speed, 4 muffled, 1 pitch |
| `speech_08_curie_baseline.wav` | 12.66s | 12.20s | 5 | 23.7 | 24.6 | 1 speed, 4 muffled |
| `speech_09_mandela_baseline.wav` | 11.67s | 11.40s | 9 | 46.3 | 47.4 | 7 speed, 1 muffled, 1 pitch |
| `speech_10_turing_baseline.wav` | 11.88s | 11.55s | 9 | 45.5 | 46.8 | 4 speed, 3 muffled, 1 pitch, 1 energy |
| **Combined Clean Baselines** | **47.93s** | **46.55s** | **32** | **40.06 FP/min** | **41.25 FP/min** | **15 speed, 12 muffled, 3 pitch, 2 energy** |

---

### 3. Deep Root-Cause Diagnoses

#### 3.1 Root Cause 1: Spectral Centroid Gap between Synthetic DSP and Natural Speech
- **Mechanism:** In the original frozen benchmark (`scripts/generate_dataset.py`), speech audio was synthesized by adding continuous high-frequency aspiration noise across the entire recording. As a result, the voiced spectral centroid never dropped below $1760\text{ Hz}$ in healthy speech. A lowpass Butterworth filter at $750\text{ Hz}$ dropped the centroid to $\approx 320 - 430\text{ Hz}$.
- **Failure Mode on TTS:** In natural human speech and clean TTS (e.g., macOS `say`), unvoiced aspiration noise is absent during pure voiced vowels (especially low back vowels like /u/, /o/, nasals /m/, /n/, and liquids /l/). Energy is concentrated in F0 ($150-200\text{ Hz}$) and F1 ($300-600\text{ Hz}$). Consequently, normal vowels naturally dip to $350 - 520\text{ Hz}$ over $300 - 500\text{ms}$ spans.
- **Detector Vulnerability:** `core/detector.py` set `THRESH_MUFFLED_CENTROID_HZ = 550.0` with `min_duration_s = 0.40s`. Every vowel sequence lasting just 400ms was classified as muffled articulation.
- **Evidence:** On clean baselines, 12 false positives occurred on normal words ("first of all", "compete with men", "persisting unaltered"). On genuine muffled flaws, the entire phrase had low spectral centroid across voiced and unvoiced frames ($\text{mean} \le 435\text{ Hz}$), whereas normal speech had high-frequency consonants reaching $3000 - 6000\text{ Hz}$ immediately adjacent to the vowel.

#### 3.2 Root Cause 2: Alignment Refinement Rate Inflation on Short Function Words
- **Mechanism:** In `core/alignment.py`, `refine_speech_rate_with_alignment()` computed expected word duration via:
  $$\text{exp\_d} = \frac{60}{\text{BASELINE\_RATE}} \cdot (0.6 + 0.08 \cdot \text{chars})$$
  For a 2-letter word ("of", "in", "to", "we"), $\text{exp\_d} = 0.444 \cdot (0.6 + 0.16) = 0.338\text{s}$ (338 ms). For a 4-letter word ("firm", "that", "this"), $\text{exp\_d} = 0.408\text{s}$ (408 ms).
- **Failure Mode on TTS:** Connected fluent English speech articulates unstressed function words in $80 - 160\text{ ms}$. When a speaker said "firm" in 100ms, the formula computed:
  $$\text{single\_r} = 135 \cdot \frac{0.408}{0.100} = 551\text{ WPM}$$
  Capped at $320\text{ WPM}$, the frame-by-frame WPM was locked to $320.0\text{ WPM}$ across the entire word.
- **Detector Vulnerability:** When two short function words occurred consecutively ("first of all let me assert my firm belief"), a $1.7\text{s}$ span was labeled $320.0\text{ WPM}$, breaching `THRESH_FAST_WPM = 175.0` and triggering an excessive speed flaw on normal 140 WPM speech.

#### 3.3 Root Cause 3: Non-Causal Window and Syllable Cluster Granularity
- **Mechanism:** In `core/features.py`, `estimate_local_wpm()` used a $0.5\text{s}$ centered window with lookahead (`next_p > t`). In a $0.5\text{s}$ window, 2 peaks produced $(2 / 0.5) \cdot 60 / 1.45 = 165.5\text{ WPM}$, while 3 peaks produced $248.3\text{ WPM}$.
- **Failure Mode:** Any polysyllabic word with two syllable peaks spaced $150\text{ms}$ apart immediately forced local density to jump from $165.5$ to $248.3\text{ WPM}$, triggering rate alarms without sustained elevation.

#### 3.4 Root Cause 4: Pause Boundaries Classified as "Slow Speech"
- **Mechanism:** `core/detector.py` flagged slow speech when $0 < \text{wpm} \le 95.0$.
- **Failure Mode:** At the onset or tail of pauses, speech rate decays toward zero. Frames where $0 < \text{wpm} \le 95$ persisting for 400ms generated 15 spurious `slow_speech` false positives on the frozen benchmark.

---

### 4. Representative Error Cases

#### Case 1: Muffled Articulation False Positive on Clean Baseline
- **File:** `data/tts_expanded/audio/baselines/speech_10_turing_baseline.wav`
- **Interval:** $0.55\text{s} - 1.80\text{s}$ (duration: $1.25\text{s}$)
- **Transcript Slice:** "may hope that machines will eventually"
- **Observed Centroid:** $486.4\text{ Hz}$ during voiced frames of "machines"
- **Acoustic Reality:** The diphthong and nasal in "machines" have low F0/F1, but the consonants /h/, /p/, /t/, /ch/, /s/ contain energy up to $5656\text{ Hz}$ ($\text{mean centroid} = 1764.6\text{ Hz}$). The phrase was fully articulate, not muffled.

#### Case 2: Excessive Speed False Positive on Normal Baseline
- **File:** `data/tts_expanded/audio/baselines/speech_07_roosevelt_baseline.wav`
- **Interval:** $0.35\text{s} - 2.05\text{s}$ (duration: $1.70\text{s}$)
- **Transcript Slice:** "first of all let me assert my firm belief"
- **Observed Value:** $320.0\text{ WPM}$ (clamped maximum)
- **Acoustic Reality:** 9 words spoken in $2.60\text{s}$ ($207\text{ WPM}$ instantaneous raw rate, 12 syllables in $2.6\text{s} = 4.6\text{ syl/s} = 191\text{ WPM}$). Alignment formula inflated monosyllabic words "firm" (100ms), "of" (150ms) to $320\text{ WPM}$, triggering a severe flaw on ordinary delivery.

#### Case 3: False Negative Due to Fragmentation on Curie Unstable Energy
- **File:** `data/tts_expanded/audio/flawed/speech_08_curie_unstable_energy_severe.wav`
- **Ground Truth:** $5.06\text{s} - 8.24\text{s}$ ("persisting unaltered through all chemical transformations")
- **Prediction:** $5.23\text{s} - 6.50\text{s}$ (observed span: $17.7\text{ dB}$)
- **IoU:** $1.27\text{s} / 3.18\text{s} = 0.399$ ($< 0.50$, counted as FN)
- **Acoustic Reality:** Swings occurred heavily from $5.20\text{s}$ to $6.30\text{s}$. At $7.10\text{s} - 7.45\text{s}$, a natural silent breath pause occurred. Because `max_gap_s = 0.75\text{s}` did not bridge over the intra-phrase pause to subsequent syllable swings, the detector truncated at $6.50\text{s}$.

---

### 5. Architectural Remediation Plan (Phase 3)

1. **Causal Speech Rate Estimator:** Replace centered lookahead with a causal $1.0\text{s}$ historical window ($[t - 1.0, t]$) and backward-looking instantaneous inter-peak deltas.
2. **Syllable-Scaled Context Alignment:** Scale expected word durations by phonetic syllable counts without a fixed 266ms floor; weight multi-word phrase context over noisy single-word durations.
3. **Muffled Articulation Spectral Gating:** Lower `THRESH_MUFFLED_CENTROID_HZ` to $440.0\text{ Hz}$, increase minimum duration to $0.50\text{s}$, and require the mean voiced spectral centroid over the interval to be $\le 550.0\text{ Hz}$.
4. **Sustained Rate Requirement & Speaker Normalization:** Require sustained elevation ($\ge 0.45\text{s}$) with dynamic threshold $\max(180.0, \text{average\_wpm} + 20.0\text{ WPM})$.
5. **Slow Speech Voicing Floor:** Restrict slow speech detection to active voiced speech with RMS $> -32.0\text{ dBFS}$ to eliminate pause decay false positives.
6. **Energy Swing Merging:** Increase `max_gap_s` from $0.75\text{s}$ to $0.85\text{s}$ to bridge intra-phrase breath transitions.
