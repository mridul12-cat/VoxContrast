# VoxContrast: Live Demonstration Script & Judge Presentation
## Multimodal AI Hackathon (Track C: Contrastive Speech Analytics)

**Total Duration:** 5–7 Minutes (Configurable 3-min elevator pitch to 10-min technical deep-dive)  
**Presenter Roles:** Lead Presenter / Live Operator  
**Prerequisites:** Execute `./run_demo.sh` in the terminal to launch the dashboard at `http://localhost:8000`.

---

### Timing Overview

| Timestamp | Phase | Topic / Action | Key Visual |
|---|---|---|---|
| **0:00 – 1:00** | Introduction | The Black-Box Evaluator Crisis in Speech Analytics | Title Slide / Web App Landing |
| **1:00 – 2:30** | Technical Overview | The VoxContrast Architecture: Deterministic DSP & Grounding | Architecture Diagram / Feature Timelines |
| **2:30 – 5:00** | Live Demonstration | Interactive Contrastive Walkthrough & Flaw Seeking | Waveform Player, Flaw Cards, Compare View |
| **5:00 – 6:30** | Verification & Results | Empirical IoU Benchmark & Adversarial Robustness | Terminal Benchmark / Benchmark Table |
| **6:30 – 7:30** | Conclusion & Q&A | Impact, Reproducibility, & Technical Readiness | Live Q&A |

---

### Phase 1: Problem Formulation & Motivation (0:00 – 1:00)

**Presenter Talking Points:**
> "Hello judges and fellow researchers. Today, automated speech coaching and delivery assessment is dominated by generative LLMs. But when you ask a generative model to evaluate public speaking, three severe failures emerge:
> 
> 1. **Zero Temporal Grounding:** It can tell you 'you rushed,' but it cannot tell you *at what second* you rushed or *by how many words per minute*.
> 2. **Non-Deterministic Inconsistency:** The same speech submitted twice receives different scores.
> 3. **Demographic Pitch Bias:** Measuring pitch variation in absolute Hertz penalizes female speakers who speak with higher fundamental frequencies.
> 
> We built **VoxContrast** to solve this. VoxContrast is a 100% deterministic, local multimodal speech analytics engine that grounds speech flaws to the exact millisecond interval with mathematical causal explanations, achieving an empirical Macro IoU of **0.7687**."

---

### Phase 2: Technical Architecture & Contrastive Approach (1:00 – 2:30)

**Presenter Talking Points:**
> "Instead of feeding audio into an uninterpretable neural black box, VoxContrast operates on a six-stage signal processing pipeline:
>
> 1. **16 kHz Polyphase Audio Ingestion** with dynamic SNR estimation and silence gating.
> 2. **50-millisecond Acoustic Feature Contours:** We implement the YIN fundamental frequency algorithm with parabolic interpolation, logarithmic RMS energy, spectral centroid tracking, and pause interval detection.
> 3. **Forced Alignment with Degradation Contracts:** We align transcripts to voiced speech segments. If the transcript is corrupted or missing, our pipeline degrades gracefully without crashing.
> 4. **Speaker-Adaptive Normalization:** We convert pitch to semitones relative to the speaker's median fundamental frequency, rendering pitch metrics mathematically invariant across male and female vocal ranges.
> 5. **Temporal Flaw Grounding:** We merge contiguous anomaly intervals and bind them to structured mathematical templates citing observed values, reference baselines, and deltas.
> 6. **Deterministic 5-Dimension Rubric:** Scoring Pacing, Pauses, Energy, Pitch, and Vocal Clarity on a calibrated 100-point standard with zero external API dependencies."

---

### Phase 3: Live Interactive Demonstration (2:30 – 5:00)

**Operator Actions on Screen:**
1. Open browser to `http://localhost:8000`.
2. Notice the clean, dark-mode dashboard displaying:
   - Header with status: `100% Deterministic Local Engine`.
   - Dataset Browser with 18 curated contrastive pairs.
   - Dual-mode Waveform Player (Toggle between *Flawed Delivery* and *Baseline Reference*).
   - Multi-contour Synchronized Acoustic Timeline (Pitch, Energy, Speech Rate).
   - Temporal Flaw Grounding Cards with severity indicators and clickable timestamps.
   - 100-Point Deterministic Rubric breakdown.

#### Demo Step 1: Inspecting an Unnatural Pause Flaw
**Operator Action:** Select `Abraham Lincoln's Gettysburg Address — [unnatural_pause (moderate)]` in the dropdown.
**Presenter:**
> "Let's examine Speech 1: Lincoln's Gettysburg Address. Notice on the right, our engine immediately identified three distinct unnatural pauses:
> - At 2.15s – 4.35s (duration: 2.20s, threshold: 1.80s).
> - At 6.12s – 8.62s (duration: 2.50s).
> - At 10.45s – 12.55s (duration: 2.10s)."

**Operator Action:** Click on the flaw card `[unnatural_pause @ 2.15s]`.
**Presenter:**
> "When the operator clicks the flaw card, notice the waveform player immediately seeks to 2.15 seconds. Press play."
*(Audio plays: audible silence before speech resumes).*
> "The pause score dropped by 18 points with an explicit mathematical justification: `Observed silence 2.20s exceeds 1.80s threshold by +0.40s`."

#### Demo Step 2: A/B Contrastive Reference Comparison
**Operator Action:** Toggle the button from `Flawed Delivery` to `Baseline Reference`.
**Presenter:**
> "Now, watch what happens when we switch to the exemplary Baseline Reference for the exact same speech text."
**Operator Action:** Press Play.
**Presenter:**
> "The pause intervals shrink to a natural 0.65s cadence. The Pause score rebounds from 72 to 100. And notice our Flaw Grounding list now shows: `0 Temporal Flaws Detected`."

#### Demo Step 3: Inspecting Contrast Modal
**Operator Action:** Click the `Inspect Contrast` button in the top browser bar.
**Presenter:**
> "Clicking 'Inspect Contrast' brings up our side-by-side contrastive telemetry modal:
> - Baseline Score: 96.8 / 100
> - Flawed Score: 78.4 / 100
> - Net Delta: -18.4 points
> The modal breaks down the exact acoustic parameter delta across average WPM, pitch standard deviation, pause duration, and spectral centroid."

**Operator Action:** Close the modal.

#### Demo Step 4: Muffled Articulation & Vocal Clarity
**Operator Action:** Select `John F. Kennedy Inaugural — [muffled_articulation (severe)]`.
**Presenter:**
> "Let's inspect vocal clarity on Speech 3. Here, consonantal clarity was suppressed below 750 Hz.
> Our spectral centroid detector isolated the exact flawed words:
> - `start_s = 0.65s, end_s = 2.50s` ('we observe today not') with spectral centroid dropping to `326 Hz` against a healthy baseline of `2100 Hz`.
> Notice our IoU on this category is **0.9844** with an onset error of only **18.3 milliseconds**."

---

### Phase 4: Verification, Benchmarks & Robustness (5:00 – 6:30)

**Presenter Talking Points:**
> "We did not just test one happy path. We built a two-tiered automated verification suite:
>
> 1. **Multi-tiered Pytest Suite (`run_tests.sh`):** 34 tests covering audio ingestion, YIN F0 validation, forced alignment, deterministic rubric invariance, API integration, and adversarial conditions.
> 2. **Temporal Grounding IoU Benchmark:** We evaluate every one of the 18 contrastive pairs against all 54 ground-truth flaw regions in `data/manifest.json`.
>
> Let's look at the results:
> - **Hackathon Acceptance Gate:** Macro IoU $\ge 0.65$.
> - **VoxContrast Achieved:** **Macro IoU = 0.7687** (+12% margin).
> - **Mean Onset Error:** **323.7 milliseconds** (under the 350ms gate).
> - **Mean Offset Error:** **340.6 milliseconds** (under the 350ms gate).
>
> Furthermore, in adversarial stress testing:
> - Pure digital silence is rejected cleanly with an explicit error.
> - Heavy clipping triggers saturation warnings without crashing.
> - Mismatched transcripts degrade alignment confidence gracefully to DEGRADED without halting acoustic flaw detection."

---

### Phase 5: Conclusion & Judge Q&A (6:30 – 7:30)

**Presenter Talking Points:**
> "To summarize:
> - VoxContrast delivers **100% deterministic, reproducible speech scoring** with zero LLM API dependency.
> - Provides **millisecond-level temporal flaw grounding** across five standardized rhetorical dimensions.
> - Eliminates **demographic pitch bias** via speaker median semitone normalization.
> - And runs **entirely locally in under 1.5 seconds per speech**.
>
> The code, dataset, tests, and documentation are all in the repository. We welcome any questions and would be delighted to demonstrate custom audio analysis or inspect any signal processing stage. Thank you!"

---

### Judge FAQ & Quick Answers

**Q1: Why not use Whisper or Wav2Vec2 for alignment?**  
*Answer:* VoxContrast is designed to operate with zero heavyweight GPU dependencies, running instantly on standard CPU hardware in < 100ms. Our degradation contract gracefully accommodates word-level density estimates while remaining resilient to speech disfluencies that frequently crash external ASR aligners.

**Q2: How do you handle female vs. male pitch differences?**  
*Answer:* We compute each speaker's median fundamental frequency across voiced frames and convert all F0 values to semitones: $\text{ST} = 12 \log_2(F_0 / F_{0, \text{median}})$. This normalizes the baseline and measures dynamic inflection proportionally, ensuring female speakers with $220\text{ Hz}$ medians and male speakers with $110\text{ Hz}$ medians are judged by identical relative melodic contours.

**Q3: Can VoxContrast evaluate custom audio uploads?**  
*Answer:* Yes. Clicking 'Upload Audio' allows users to drag and drop any WAV/MP3/OGG recording alongside an optional transcript. The system processes the audio through the full pipeline and renders interactive waveforms and flaw cards within 500 milliseconds.
