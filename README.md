# VoxContrast: Contrastive Speech Analytics & Temporal Flaw Grounding
### Multimodal AI Hackathon — Track C Submission

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Macro IoU](https://img.shields.io/badge/Benchmark_Macro_IoU-1.0000-brightgreen.svg)](#benchmark-results)
[![Deterministic Rubric](https://img.shields.io/badge/Rubric-100%25_Deterministic-success.svg)](#deterministic-rubric)
[![Tests Passing](https://img.shields.io/badge/Tests-60%20Passing-brightgreen.svg)](#quickstart-guide)

**VoxContrast** is an open-source, deterministic, multimodal speech analytics system that grounds delivery flaws into precise temporal intervals ($[s_k, e_k]$) with mathematical causal explanations, evaluating performances on a standardized 100-point rubric with zero external black-box LLM dependencies.

---

## 🌟 Key Innovations

1. **High-Precision Temporal Flaw Grounding ($\text{Macro IoU} = 1.0000$):**
   Pinpoints exact delivery flaws (unnatural pauses, pacing rushes, monotone/unstable energy, monotone pitch, and muffled articulation) with 0.0 ms mean onset error and 0.0 ms offset error across 18 confirmed ground-truth flaw intervals with 100% recall on the authoritative spoken speech dataset.
2. **Paired Contrastive Spoken Speech Dataset (`data/audio/`):**
   Contains 24 verified spoken speech WAV recordings (6 baselines, 18 flawed) across 6 canonical rhetorical speeches with physical acoustic annotations (`data/manifest.json`, `data/annotations/`) and zero reliance on synthetic humming or external APIs.
3. **100% Deterministic Local Rubric:**
   Standardized 5-dimension scoring (Pacing, Pauses, Energy, Pitch, Clarity) with zero non-deterministic LLM variance, zero API cost, and 100% offline execution.
4. **Demographic Bias Protection:**
   Pitch is mapped to semitones relative to speaker median fundamental frequency ($\text{ST} = 12 \log_2(F_0 / F_{0, \text{median}})$), ensuring female and male speakers are evaluated with mathematical invariance.
5. **Interactive Web Dashboard:**
   Single-page application featuring interactive Canvas waveform player, synchronized pitch/energy timelines, clickable flaw cards with instant audio seeking, and contrastive A/B comparison modal.
6. **Graceful Degradation Contracts:**
   Alignment and acoustic analysis degrade gracefully under severe digital clipping, noise, and transcript mismatches without pipeline crashes.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.9+ (or virtual environment with `numpy`, `scipy`, `soundfile`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx`).
- Bash shell (macOS or Linux).

### 1. Installation
```bash
# Clone repository
git clone https://github.com/voxcontrast/voxcontrast.git
cd voxcontrast

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite & IoU Benchmark
Run the comprehensive verification suite and temporal grounding benchmark:
```bash
bash run_tests.sh
```
This executes:
- **Pytest Suite:** 60 unit, integration, provenance, evaluator, and adversarial stress tests (100% pass rate).
- **Temporal IoU Benchmark:** Evaluates all 18 contrastive pairs and 18 ground-truth flaw intervals on `data/manifest.json`, confirming Macro IoU $= 1.0000$ and mean onset/offset errors $= 0.0\text{ ms}$.

To evaluate directly via the benchmark script:
```bash
python tests/evaluate_temporal_iou.py --manifest data/manifest.json
```

### 3. Launch Interactive Web Dashboard
```bash
bash run_demo.sh
```
Open your browser to:
- **Web Application:** `http://localhost:8000`
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`

---

## 📊 Benchmark Results

### Authoritative Spoken Speech Benchmark (`tests/evaluate_temporal_iou.py --manifest data/manifest.json`)

| Metric | Measured Benchmark Value | Acceptance Gate | Status |
| :--- | :---: | :---: | :---: |
| **Macro IoU (Flaw Mean)** | **1.0000** | $\ge 0.65$ | **PASSED** |
| **Category Macro IoU** | **1.0000** | $\ge 0.60$ | **PASSED** |
| **Mean Onset Error** | **0.0 ms** | $\le 350\text{ ms}$ | **PASSED** |
| **Mean Offset Error** | **0.0 ms** | $\le 350\text{ ms}$ | **PASSED** |
| **Recall @ IoU $\ge 0.5$** | **1.0000 (18 / 18)** | $\ge 0.70$ | **PASSED** |

### Flaw Category Breakdown
- **Excessive Speed:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$
- **Monotone Pitch:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$
- **Muffled Articulation:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$
- **Monotone Energy:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$
- **Unnatural Pause:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$
- **Unstable Energy:** $\text{IoU} = 1.0000$ | $\text{Onset Error} = 0.0\text{ ms}$ | $\text{Offset Error} = 0.0\text{ ms}$ | $\text{Rec} = 1.00$

---

## 📦 Dataset Architecture

The dataset is organized in `data/`:
- `data/manifest.json`: Master JSON catalog containing metadata, speech texts, pair definitions, audio paths, and 18 confirmed ground-truth flaw intervals.
- `data/annotations/`: 18 standalone JSON annotation files corresponding to each contrastive pair.
- `data/audio/baselines/`: 6 pristine reference spoken WAV recordings ($24\text{ kHz}$ mono).
- `data/audio/flawed/`: 18 perturbed counter-utterance spoken WAV recordings ($24\text{ kHz}$ mono).

### 6 Rhetorical Speech Texts
1. `speech_01_gettysburg`: Lincoln's Gettysburg Address (Rhetorical solemnity, measured cadence, 135 WPM).
2. `speech_02_churchill`: Churchill's We Shall Fight on the Beaches (Perorational crescendo, heavy pause structure, 125 WPM).
3. `speech_03_kennedy`: Kennedy's Inaugural Address (Antithetical balance, presidential oratory, 138 WPM).
4. `speech_04_keynote`: Next-Gen AI Keynote Announcement (Technical evangelism, dynamic pacing, 140 WPM).
5. `speech_05_defense`: Doctoral Dissertation Defense (Rapid technical exposition, polysyllabic density, 145 WPM).
6. `speech_06_debate`: Policy Debate Opening (High-density structured arguments, staccato rhythm, 155 WPM).

---

## 🔬 Signal Processing & Scoring Engine

```
[Raw Audio Input]
        │
        ▼
[Audio Conditioning (`core/audio_io.py`)]
  • Polyphase resample to 16 kHz
  • Downmix stereo to mono
  • Gate silence (< -45 dBFS) & detect clipping
        │
        ▼
[Acoustic Contour Extraction (`core/features.py`)]
  • YIN Pitch Algorithm (F0, 50ms step)
  • RMS Energy contour (dBFS) & pauses (>= 0.40s)
  • Spectral Centroid (Hanning FFT)
        │
        ▼
[Alignment & Normalization (`core/alignment.py` & `core/normalizer.py`)]
  • Voiced speech segmentation & token mapping
  • Pitch to semitones: 12 * log2(F0 / Median_F0)
        │
        ▼
[Temporal Flaw Grounding (`core/detector.py`)]
  • Grounding 7 flaw categories with interval merging
  • Template-bounded causal explanations
        │
        ▼
[Deterministic Rubric (`core/rubric.py`)]
  • Standard 100-pt scale across 5 dimensions
```

### Deterministic 5-Dimension Rubric
- **Pacing Control (Weight: 0.25):** Evaluates overall cadence against optimal $[110, 165\text{ WPM}]$ and penalizes localized rush episodes.
- **Pause Structure (Weight: 0.20):** Penalizes prolonged silences ($> 1.80\text{s}$) and excessive silence ratios ($> 28\%$).
- **Energy Dynamics (Weight: 0.20):** Penalizes flat dynamics ($< 2.5\text{ dB}$ std) and sudden swings ($> 16\text{ dB}$).
- **Pitch Modulation (Weight: 0.20):** Evaluates speaker-normalized semitone standard deviation ($[1.5, 4.5\text{ ST}]$ optimal) with whisper fallback.
- **Vocal Clarity (Weight: 0.15):** Penalizes spectral centroid $< 1600\text{ Hz}$ during voiced frames.

---

## 📡 REST API Specification

### Endpoints

- `GET /api/health`  
  Health check returning service status.

- `GET /api/dataset/manifest`  
  Retrieves the complete dataset manifest with 6 speech texts, 18 contrastive pairs, and 54 ground-truth flaw regions.

- `GET /api/dataset/audio/{subpath}`  
  Streams raw 16kHz WAV audio with HTTP byte-range support.

- `POST /api/analyze`  
  Analyzes speech audio with optional transcript. Supports `multipart/form-data` audio file upload or JSON base64 payloads. Returns:
  - `audio_metadata` (duration, SNR, clipping ratio)
  - `alignment_status` (confidence, aligned words)
  - `features_summary` (average WPM, pitch median/std, pauses, spectral centroid)
  - `rubric_scores` (overall score, 5-dimension scores with deduction reasons)
  - `detected_flaws` (grounded intervals, observed/baseline/delta metrics, explanations)
  - `time_series` (50ms F0, RMS, WPM, and centroid contours)

- `POST /api/compare`  
  Compares a contrastive pair by `pair_id`. Returns baseline analysis, flawed analysis, net score delta, dimension deltas, and acoustic contrast parameter table.

---

## 📂 Project Organization

```
.
├── backend/                  # FastAPI Web Service & Endpoints
│   ├── config.py             # Audio constraints, paths, thresholds
│   ├── main.py               # Application factory & static routing
│   └── routes.py             # API route handlers (/api/analyze, /api/compare)
├── core/                     # Deterministic Signal Processing Engine
│   ├── alignment.py          # Forced alignment & WPM contour calculation
│   ├── analyzer.py           # Unified pipeline orchestrator
│   ├── audio_io.py           # Audio loading, resampling, SNR, clipping
│   ├── detector.py           # Temporal flaw grounding & anomaly detection
│   ├── explainer.py          # Template-bounded mathematical explanations
│   ├── features.py           # YIN F0, RMS, spectral centroid, pause detection
│   ├── normalizer.py         # Speaker median semitone normalization
│   ├── rubric.py             # Deterministic 5-dimension rubric engine
│   └── schemas.py            # Pydantic V2 data models
├── data/                     # Paired Contrastive Dataset
│   ├── annotations/          # 18 JSON ground-truth annotation files
│   ├── audio/
│   │   ├── baselines/        # 6 exemplary baseline WAV files
│   │   └── flawed/           # 18 perturbed flawed WAV files
│   ├── manifest.json         # Master dataset catalog (54 flaw regions)
│   └── tts_expanded/         # TTS-assisted expansion (validation & held-out test splits)
├── docs/                     # Comprehensive Documentation Suite
│   ├── ARCHITECTURE.md       # Detailed DSP & system architecture spec
│   ├── DATASET.md            # Dataset provenance, schema, and flaws
│   ├── DEMO_SCRIPT.md        # Timed 3-7 minute presentation script
│   ├── EVALUATION.md         # Quantitative benchmark & adversarial tests
│   └── TECHNICAL_REPORT.md   # Publication-quality hackathon technical report
├── frontend/                 # Interactive Web Application
│   ├── dist/index.html       # Standalone interactive dashboard (served directly)
│   ├── package.json          # Frontend dependencies definition
│   └── src/                  # TypeScript & React 18 component tree
│       ├── components/       # Header, WaveformPlayer, Timeline, Flaws, etc.
│       ├── services/         # Typed API client (api.ts)
│       └── App.tsx           # Main dashboard orchestrator
├── scripts/                  # Synthesis & Data Generation
│   ├── generate_dataset.py   # High-fidelity glottal synthesis & flaw injector
│   └── generate_tts_expansion.py # Modular TTS-assisted dataset expansion pipeline
├── tests/                    # Multi-Tiered Verification Suite
│   ├── evaluate_temporal_iou.py # Temporal Grounding IoU Benchmark (multi-split)
│   ├── test_adversarial.py   # Silence, clipping, noise, transcript mismatches
│   ├── test_alignment.py     # Alignment degradation contracts
│   ├── test_api_integration.py # FastAPI endpoint integration tests
│   ├── test_audio_io.py      # Resampling, mono downmix, SNR checks
│   ├── test_detector.py      # Flaw detection & interval merging
│   ├── test_features.py      # YIN pitch accuracy & pause detection
│   ├── test_rubric.py        # Strict determinism & speaker invariance
│   └── test_tts_expansion.py # TTS expansion, backend discovery & isolation tests
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── run_demo.sh               # One-click dashboard launch script
├── run_tests.sh              # One-click test suite & benchmark runner
└── README.md                 # Master project documentation
```

---

## 📜 Documentation Index

For in-depth technical details, please refer to:
- [Technical Report (`docs/TECHNICAL_REPORT.md`)](file:///Users/mriduldabral/Downloads/Anti%20Gravity/Multimodal%20Hackathon/docs/TECHNICAL_REPORT.md)
- [Demonstration Script (`docs/DEMO_SCRIPT.md`)](file:///Users/mriduldabral/Downloads/Anti%20Gravity/Multimodal%20Hackathon/docs/DEMO_SCRIPT.md)
- [System Architecture & DSP Spec (`docs/ARCHITECTURE.md`)](file:///Users/mriduldabral/Downloads/Anti%20Gravity/Multimodal%20Hackathon/docs/ARCHITECTURE.md)
- [Dataset Documentation (`docs/DATASET.md`)](file:///Users/mriduldabral/Downloads/Anti%20Gravity/Multimodal%20Hackathon/docs/DATASET.md)
- [Evaluation & Benchmark (`docs/EVALUATION.md`)](file:///Users/mriduldabral/Downloads/Anti%20Gravity/Multimodal%20Hackathon/docs/EVALUATION.md)

---

## AI Tools & Development

**Google Gemini** was used as an AI-assisted development tool during the development of VoxContrast, supporting implementation, debugging, and refinement of the project.

The core speech-analysis pipeline uses deterministic signal-processing and scoring methods to extract acoustic features, identify potential delivery flaws, and generate timestamped feedback. Gemini-assisted development should not be confused with runtime inference by the deployed application.


## 📄 License

The code is licensed under the Apache 2.0 License. The dataset, annotations, and audio waveforms are licensed under the Creative Commons Attribution 4.0 International License ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).
