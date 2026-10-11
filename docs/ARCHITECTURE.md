# VoxContrast: System Architecture & DSP Specification
## Contrastive Speech Analytics & Temporal Flaw Grounding

---

### 1. Architectural Overview

VoxContrast is engineered as a zero-cloud-dependency, 100% deterministic speech analytics engine. The system extracts acoustic feature contours at a uniform 50ms time step ($20\text{ Hz}$ sampling rate), aligns transcript tokens with explicit graceful degradation contracts, grounds delivery flaws into discrete time intervals $[s_k, e_k]$, and calculates standardized rubric scores across five dimensions.

```
                    ┌──────────────────────────────────────────────┐
                    │            Raw Speech Waveform               │
                    │         (WAV, FLAC, OGG, Base64)             │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Audio Ingestion & Normalization (`core/audio_io.py`)                           │
│ • Decode buffer via soundfile float32                                                  │
│ • Stereo-to-mono downmix: 0.5 * (L + R)                                                │
│ • Polyphase FIR Resampling to exactly 16,000 Hz                                        │
│ • Energy threshold: Reject if global RMS < -45 dBFS                                    │
│ • Telemetry: SNR estimation, clipping ratio (threshold >= 0.995)                       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Acoustic Contour Extraction at 50ms Resolution (`core/features.py`)           │
│ • Fundamental Frequency F0: YIN Algorithm (tau_min=40, tau_max=266, thresh=0.20)      │
│ • Energy Contour: Logarithmic RMS dBFS with -80 dBFS floor                             │
│ • Spectral Centroid: 1024-point Hanning window FFT center of mass                      │
│ • Pause Intervals: RMS <= -38 dBFS persisting for >= 0.40s                             │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Forced Alignment & VAD Segmentation (`core/alignment.py`)                     │
│ • Voiced speech activity segmentation                                                  │
│ • Token mapping across active speech segments                                          │
│ • Degradation contracts: SUCCESS (1.0), DEGRADED (0.15-0.70), FAILED (0.0)             │
│ • Refined instantaneous Speech Rate (WPM) contour                                      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Speaker Normalization (`core/normalizer.py`)                                  │
│ • Voiced median F0 computation                                                         │
│ • Semitone transformation: 12 * log2(F0 / Median_F0)                                   │
│ • Energy normalization relative to 90th percentile voiced RMS                          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Temporal Flaw Grounding Engine (`core/detector.py`)                           │
│ • Voiced-frame gating & localized sliding window analysis                              │
│ • Contiguous interval merging (max_gap_s = 0.25 - 0.35s, min_duration_s = 0.40s)       │
│ • Anomaly classification across 7 flaw categories                                      │
│ • Template-bounded causal explanations citing (observed, baseline, delta, threshold)   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: Deterministic 5-Dimension Rubric (`core/rubric.py`)                           │
│ • Pacing (0.25) | Pause Control (0.20) | Energy (0.20) | Pitch (0.20) | Clarity (0.15) │
│ • 100-point standard score with unvoiced pitch fallback                                │
│ • Strictly reproducible: bitwise identical scores across repeat runs                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: Presentation & Service Layer (`backend/` & `frontend/`)                       │
│ • FastAPI asynchronous endpoints: `/api/analyze`, `/api/compare`, `/api/dataset/*`     │
│ • Interactive canvas waveform player with seek-to-timestamp flaw navigation            │
│ • Synchronized multi-contour telemetry viewer (Pitch, Energy, WPM)                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Detailed DSP Module Specifications

#### 2.1 Audio Ingestion & Conditioning (`core/audio_io.py`)
- **Mono Downmixing:** Multichannel signals are summed and averaged:
  $$x_{\text{mono}}[n] = \frac{1}{C} \sum_{c=1}^C x_c[n]$$
- **Polyphase Resampling:** Signals sampled at rates other than 16,000 Hz are resampled using polyphase rational filtering (`scipy.signal.resample_poly`), preventing spectral aliasing and phase shifts:
  $$\frac{P}{Q} = \text{ReduceFraction}\left(\frac{16000}{f_{s, \text{orig}}}\right)$$
- **Digital Saturation Telemetry:** Clipped sample fraction:
  $$\text{Ratio}_{\text{clip}} = \frac{1}{N} \sum_{n=1}^N \mathbb{I}(|x[n]| \ge 0.995)$$
- **Signal-to-Noise Ratio (SNR):** Estimated from the separation between the 90th percentile voiced energy and the 10th percentile noise floor:
  $$\text{SNR}_{\text{dB}} = P_{90}(\text{RMS}_{\text{dB}}) - P_{10}(\text{RMS}_{\text{dB}})$$

---

#### 2.2 Fundamental Frequency Extraction via YIN (`core/features.py`)
The YIN pitch detection algorithm is configured with:
- Window length: $N = 1024$ samples ($64\text{ms}$ at $16\text{ kHz}$)
- Time step: $H = 800$ samples ($50\text{ms}$ at $16\text{ kHz}$)
- Minimum frequency: $F_{0, \text{min}} = 60\text{ Hz} \implies \tau_{\text{max}} = 266$
- Maximum frequency: $F_{0, \text{max}} = 400\text{ Hz} \implies \tau_{\text{min}} = 40$
- Harmonic threshold: $\theta = 0.20$

**Mathematical Formulation:**
1. Difference Function:
   $$d_t(\tau) = \sum_{j=1}^{W} \left( x[j] - x[j + \tau] \right)^2$$
2. Cumulative Mean Normalized Difference:
   $$d'_t(\tau) = \begin{cases} 1, & \tau = 0 \\ \frac{d_t(\tau)}{\frac{1}{\tau} \sum_{j=1}^\tau d_t(j)}, & \tau > 0 \end{cases}$$
3. Dip Selection & Parabolic Interpolation:
   The first local minimum satisfying $d'_t(\tau) < \theta$ is refined via 3-point parabolic interpolation:
   $$\Delta \tau = \frac{d'(\tau - 1) - d'(\tau + 1)}{2 (d'(\tau - 1) - 2 d'(\tau) + d'(\tau + 1))}$$
   $$\tau^* = \tau + \Delta \tau, \quad F_0(t) = \frac{f_s}{\tau^*}$$
Frames with energy below $-38.0\text{ dBFS}$ or failing harmonicity are marked unvoiced ($F_0 = 0.0\text{ Hz}$).

---

#### 2.3 Syllabic Energy & Pause Detection (`core/features.py`)
- **Logarithmic RMS Energy:**
  $$\text{RMS}(t) = 20 \log_{10} \left( \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[t \cdot H + n]^2} + 10^{-6} \right)$$
- **Pause Intervals:** Any continuous run of frames where $\text{RMS}(t) \le -38.0\text{ dBFS}$ lasting $\ge 0.40\text{s}$ is logged as a pause interval $P_m = [t_{\text{start}}, t_{\text{end}}]$. Pauses exceeding $1.80\text{s}$ are classified as excessive delivery hesitations.

---

#### 2.4 Spectral Centroid (`core/features.py`)
A Hanning window $w[n] = 0.5 - 0.5 \cos(2\pi n / N)$ is applied before computing the discrete Fourier transform $X[k]$. The spectral centroid (brightness/clarity) is calculated as:
$$\text{Centroid}(t) = \frac{\sum_{k=0}^{K} f_k |X[k]|}{\sum_{k=0}^{K} |X[k]|}$$
Voiced frames with centroid $< 1200\text{ Hz}$ signify muffled articulation or severe acoustic dampening.

---

#### 2.5 Forced Alignment & Speech Rate Refinement (`core/alignment.py`)
1. **Voiced Activity Intervals:** Contiguous voiced segments $V_j = [v_{j, \text{start}}, v_{j, \text{end}}]$ are identified from RMS energy above $-38\text{ dBFS}$.
2. **Plausibility & Degradation Contracts:**
   - Average speech rate: $\text{WPM}_{\text{exp}} = \frac{N_{\text{words}}}{\sum |V_j| / 60}$.
   - If $\text{WPM}_{\text{exp}} \in [45, 320]$: Status is `SUCCESS` (Confidence = $1.0$).
   - If $\text{WPM}_{\text{exp}} < 45$ or $> 320$: Status is `DEGRADED` (Confidence = $0.15 - 0.70$) with explicit reason.
   - If transcript is empty or zero voiced frames: Status is `FAILED` (Confidence = $0.0$).
3. **Instantaneous Speech Rate (WPM):** Aligned word durations and local 3-word context windows compute local WPM for each frame:
   $$\text{WPM}(t) = \frac{N_{\text{context}}}{\sum_{w \in \text{context}} \text{Duration}(w)} \times 60$$

---

#### 2.6 Speaker Normalization (`core/normalizer.py`)
To prevent demographic bias against naturally higher-pitched speakers (e.g., female vocal ranges), all pitch measurements are normalized into semitones relative to the speaker's voiced median fundamental frequency:
$$\text{ST}(t) = 12 \log_2 \left( \frac{F_0(t)}{F_{0, \text{median}}} \right)$$
- A pitch variation of $\pm 3\text{ semitones}$ represents identical musical intonation whether the median is $110\text{ Hz}$ or $240\text{ Hz}$.
- Normalized energy is referenced against the 90th percentile voiced RMS level.

---

### 3. Temporal Flaw Grounding Architecture (`core/detector.py`)

#### 3.1 Interval Detection Rules
| Dimension | Flaw Category | Mathematical Detection Condition | Window / Merging |
|---|---|---|---|
| **Pacing** | `excessive_speed` | $\text{WPM}(t) \ge 175.0\text{ WPM}$ | $\text{min\_dur} \ge 0.50\text{s}$, $\text{gap} \le 0.25\text{s}$ |
| **Pauses** | `unnatural_pause` | $\text{Duration}(P) \ge 1.80\text{s}$ | Exact silence interval |
| **Energy** | `unstable_energy` | Syllabic peak swing $\ge 16.0\text{ dB}$ | Peak-to-peak window $\pm 0.50\text{s}$ |
| **Energy** | `monotone_energy` | Voiced RMS $\text{std} < 1.50\text{ dB}$ | $1.2\text{s}$ window, $\text{min\_dur} \ge 1.20\text{s}$ |
| **Pitch** | `monotone_pitch` | Voiced semitone $\text{std} < 0.65\text{ ST}$ | $1.2\text{s}$ window, $\text{min\_dur} \ge 1.40\text{s}$ |
| **Pitch** | `pitch_irregularity` | Voiced semitone $\text{std} > 2.80\text{ ST}$ | $0.6\text{s}$ window, $\text{gap} \le 0.35\text{s}$ |
| **Clarity** | `muffled_articulation`| Voiced centroid $< 1200\text{ Hz}$ | Frame-level, $\text{gap} \le 0.35\text{s}, \ge 0.40\text{s}$ |

#### 3.2 Interval Merging Algorithm
Given raw detection intervals $I = \{(s_1, e_1), (s_2, e_2), \dots\}$:
1. Sort chronologically by onset $s_i$.
2. Merge adjacent intervals if $s_{i+1} - e_i \le \text{max\_gap\_s}$.
3. Discard merged intervals where total duration $e_{\text{merged}} - s_{\text{merged}} < \text{min\_duration\_s}$.

---

### 4. Deterministic 5-Dimension Rubric Specification (`core/rubric.py`)

Each dimension starts at a maximum score and incurs deterministic deductions based on acoustic telemetry:

#### 4.1 Dimension 1: Pacing Control ($w = 0.25$, Max = 25 pts)
- **Base Score:** 25.0
- **Rate Penalty:** Deducts up to 10 points if average WPM falls outside optimal $[110, 165]$ range:
  $$\text{Ded}_{\text{rate}} = \min(10.0, |\text{WPM}_{\text{avg}} - 137.5| \times 0.20)$$
- **Flaw Penalty:** Deducts 3.5 points per localized `excessive_speed` episode (max 12 pts).

#### 4.2 Dimension 2: Pause Structure ($w = 0.20$, Max = 20 pts)
- **Base Score:** 20.0
- **Prolonged Pause Penalty:** Deducts 4.0 points per `unnatural_pause` (duration $\ge 1.80\text{s}$, max 12 pts).
- **Silence Ratio Penalty:** Deducts if total pause time exceeds $28\%$ of total speech duration:
  $$\text{Ded}_{\text{silence}} = \min(8.0, \max(0.0, \text{Ratio}_{\text{silence}} - 0.28) \times 35.0)$$

#### 4.3 Dimension 3: Energy Dynamics ($w = 0.20$, Max = 20 pts)
- **Base Score:** 20.0
- **Flat Dynamics Penalty:** Deducts up to 8 points if voiced RMS standard deviation $< 2.5\text{ dB}$:
  $$\text{Ded}_{\text{flat}} = \max(0.0, (2.5 - \text{RMS}_{\text{std}}) \times 3.0)$$
- **Instability Penalty:** Deducts 3.0 points per `unstable_energy` episode (max 10 pts).

#### 4.4 Dimension 4: Pitch Modulation ($w = 0.20$, Max = 20 pts)
- **Base Score:** 20.0
- **Voiced Dynamic Range:** Optimal range for semitone standard deviation is $[1.5, 4.5\text{ ST}]$.
  - Monotone deduction: $\max(0.0, (1.5 - \text{ST}_{\text{std}}) \times 5.0)$
  - Erratic deduction: $\max(0.0, (\text{ST}_{\text{std}} - 4.5) \times 3.0)$
- **Unvoiced Whisper Fallback:** If voiced frame ratio $< 15\%$, pitch weight is automatically rescaled to Energy and Clarity dimensions to prevent penalizing naturally whispered delivery.

#### 4.5 Dimension 5: Vocal Clarity ($w = 0.15$, Max = 15 pts)
- **Base Score:** 15.0
- **Centroid Penalty:** Deducts up to 7 points if mean voiced spectral centroid $< 1600\text{ Hz}$:
  $$\text{Ded}_{\text{cent}} = \max(0.0, (1600.0 - \text{Centroid}_{\text{mean}}) \times 0.007)$$
- **Muffled Flaw Penalty:** Deducts 3.5 points per `muffled_articulation` flaw episode.

**Standard 100-Point Overall Score:**
$$\text{Overall Score} = \sum_{d=1}^5 \max(0.0, \text{Score}_d)$$

---

### 5. API and Web Service Architecture

The backend (`backend/main.py`) exposes high-performance asynchronous endpoints:
- `GET /api/health`: Service health and version status.
- `GET /api/dataset/manifest`: Returns dataset metadata, speech texts, and 18 contrastive pairs.
- `GET /api/dataset/audio/{subpath}`: Byte-range audio streaming for baseline and flawed WAV files.
- `POST /api/analyze`: Multipart file upload or JSON base64 speech analysis.
- `POST /api/compare`: Form-encoded pair ID comparison yielding baseline analysis, flawed analysis, net score delta, and parameter contrast summary.
- `GET /`: Static mount serving the modern interactive web application (`frontend/dist/index.html`).

---
*VoxContrast DSP Specification — October 2026*
