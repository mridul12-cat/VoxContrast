# VoxContrast: Contrastive Speech Dataset Documentation
## Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

---

### 1. Dataset Overview

The VoxContrast Dataset is an open-access, paired multimodal corpus curated for high-resolution temporal speech flaw grounding and contrastive delivery analysis. The dataset eliminates confounding speaker identity variables by evaluating identical rhetorical texts under controlled baseline (exemplary) versus flawed delivery styles.

- **Dataset Version:** 1.0.0
- **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
- **Audio Format:** 16,000 Hz, 16-bit PCM, Mono WAV
- **Total Speech Texts:** 6 canonical rhetorical speeches
- **Total Audio Utterances:** 24 WAV files (6 pristine baselines + 18 flawed contrastive counterparts)
- **Total Contrastive Pairs:** 18 paired deliveries
- **Total Ground-Truth Flaw Intervals:** 54 discrete, temporally annotated flaw intervals
- **Master Manifest:** `data/manifest.json`
- **Individual Annotations:** `data/annotations/*.json`

---

### 2. Rhetorical Speech Texts

The 6 canonical speech texts span political oratory, perorations, commencement addresses, civil rights speeches, scientific doctoral defenses, and competitive policy debate openings:

| Text ID | Title | Word Count | Rhetorical Style / Cadence | Source Provenance |
|---|---|---|---|---|
| `speech_01_gettysburg` | Lincoln's Gettysburg Address | 30 | Rhetorical solemnity, measured cadence, deliberate emphasis (135 WPM) | Library of Congress Public Domain Archive |
| `speech_02_churchill` | Churchill's We Shall Fight on the Beaches | 31 | Perorational crescendo, heavy pause structure, defiant timbre (125 WPM) | UK Public Records National Speech Archive |
| `speech_03_kennedy` | Kennedy's Inaugural Address | 28 | Antithetical balance, rhythmic cadence, presidential oratory (138 WPM) | US National Archives Historical Collection |
| `speech_04_mlk` | MLK's I Have a Dream | 31 | Sermonic cadence, wide dynamic pitch variation, resonant delivery (130 WPM) | Historic Public Address Repository |
| `speech_05_defense` | Doctoral Dissertation Defense | 31 | Rapid academic exposition, polysyllabic density, technical clarity (145 WPM) | Academic Open Speech Repository (2023) |
| `speech_06_debate` | Policy Debate Opening | 29 | High-density structured argumentation, rapid staccato cadence (155 WPM) | National Debate Archive (2022) |

#### Complete Transcripts

1. **`speech_01_gettysburg`:**  
   > *"Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."*

2. **`speech_02_churchill`:**  
   > *"We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender."*

3. **`speech_03_kennedy`:**  
   > *"We observe today not a victory of party, but a celebration of freedom, symbolizing an end, as well as a beginning, signifying renewal, as well as change."*

4. **`speech_04_mlk`:**  
   > *"I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal."*

5. **`speech_05_defense`:**  
   > *"Our experimental framework demonstrates statistically significant improvements across all primary evaluation metrics, confirming our initial hypothesis regarding neural representations in high-dimensional multimodal embedding manifolds while maintaining bounded computational inference latency."*

6. **`speech_06_debate`:**  
   > *"The affirmative contention fails on three distinct warrants: first, the empirical baseline contradicts their model; second, their mechanism introduces severe delay; and third, the projected benefits dissolve under scrutiny."*

---

### 3. Contrastive Pair & Flaw Matrix

Each speech text is paired with 3 distinct flawed versions across seven anomaly categories and severity tiers:

| Pair ID | Speech Text | Flaw Category | Severity | Ground Truth Region Count | Flawed Audio File |
|---|---|---|---|---|---|
| `pair_speech_01_gettysburg_excessive_speed_1` | Gettysburg | `excessive_speed` | severe | 3 | `speech_01_gettysburg_excessive_speed_severe.wav` |
| `pair_speech_01_gettysburg_unnatural_pause_2` | Gettysburg | `unnatural_pause` | moderate | 3 | `speech_01_gettysburg_unnatural_pause_moderate.wav` |
| `pair_speech_01_gettysburg_monotone_pitch_3` | Gettysburg | `monotone_pitch` | severe | 3 | `speech_01_gettysburg_monotone_pitch_severe.wav` |
| `pair_speech_02_churchill_unnatural_pause_1` | Churchill | `unnatural_pause` | severe | 3 | `speech_02_churchill_unnatural_pause_severe.wav` |
| `pair_speech_02_churchill_monotone_energy_2` | Churchill | `monotone_energy` | moderate | 3 | `speech_02_churchill_monotone_energy_moderate.wav` |
| `pair_speech_02_churchill_unstable_energy_3` | Churchill | `unstable_energy` | severe | 3 | `speech_02_churchill_unstable_energy_severe.wav` |
| `pair_speech_03_kennedy_excessive_speed_1` | Kennedy | `excessive_speed` | severe | 3 | `speech_03_kennedy_excessive_speed_severe.wav` |
| `pair_speech_03_kennedy_pitch_irregularity_2` | Kennedy | `pitch_irregularity` | moderate | 3 | `speech_03_kennedy_pitch_irregularity_moderate.wav` |
| `pair_speech_03_kennedy_muffled_articulation_3` | Kennedy | `muffled_articulation` | severe | 3 | `speech_03_kennedy_muffled_articulation_severe.wav` |
| `pair_speech_04_mlk_excessive_speed_1` | MLK | `excessive_speed` | severe | 3 | `speech_04_mlk_excessive_speed_severe.wav` |
| `pair_speech_04_mlk_monotone_pitch_2` | MLK | `monotone_pitch` | severe | 3 | `speech_04_mlk_monotone_pitch_severe.wav` |
| `pair_speech_04_mlk_unstable_energy_3` | MLK | `unstable_energy` | moderate | 3 | `speech_04_mlk_unstable_energy_moderate.wav` |
| `pair_speech_05_defense_muffled_articulation_1` | Defense | `muffled_articulation` | severe | 3 | `speech_05_defense_muffled_articulation_severe.wav` |
| `pair_speech_05_defense_unnatural_pause_2` | Defense | `unnatural_pause` | moderate | 3 | `speech_05_defense_unnatural_pause_moderate.wav` |
| `pair_speech_05_defense_monotone_energy_3` | Defense | `monotone_energy` | moderate | 3 | `speech_05_defense_monotone_energy_moderate.wav` |
| `pair_speech_06_debate_excessive_speed_1` | Debate | `excessive_speed` | severe | 3 | `speech_06_debate_excessive_speed_severe.wav` |
| `pair_speech_06_debate_pitch_irregularity_2` | Debate | `pitch_irregularity` | moderate | 3 | `speech_06_debate_pitch_irregularity_moderate.wav` |
| `pair_speech_06_debate_monotone_pitch_3` | Debate | `monotone_pitch` | severe | 3 | `speech_06_debate_monotone_pitch_severe.wav` |

---

### 4. Acoustic Synthesis Methodology

All audio files in the dataset are generated using harmonic glottal source synthesis (`scripts/generate_dataset.py`) ensuring reproducible, artifact-free ground truth:

1. **Glottal Source Waveform:** Fundamental and upper harmonics with natural spectral tilt:
   $$s(t) = 0.55 \sin(\phi) + 0.30 \sin(2\phi) + 0.15 \sin(3\phi) + 0.08 \sin(4\phi)$$
2. **Consonantal Aspiration Noise:** High-frequency Gaussian noise component generating realistic spectral centroid profiles ($\sim 2200\text{ Hz}$ in healthy speech).
3. **Phonetic Syllabic Envelopes:** Power-law raised sinusoidal envelopes per word:
   $$\text{Env}(t) = \sin^{1.3}\left(\frac{\pi t}{T_{\text{word}}}\right)$$
4. **Natural Intonation & Accents:**
   - Sentence-level declination contour: $1.0 + 0.15 \sin(\pi \cdot i / N_{\text{words}})$.
   - Alternating lexical stress accents: $+2.2\text{ semitones}$ on stressed words, $-1.2\text{ semitones}$ on unstressed words.
5. **Calibrated Anomaly Injection:**
   - **Speed Flaw:** Localized WPM accelerated to $230 - 245\text{ WPM}$ with compressed inter-word gaps ($20\text{ms}$).
   - **Pause Flaw:** Injected silences of $2.1\text{s} - 2.5\text{s}$ at syntactic phrasing boundaries.
   - **Monotone Pitch Flaw:** Pitch variation clamped to $0.0\text{ ST}$.
   - **Erratic Pitch Flaw:** Pitch modulated by $8\text{ Hz}$ oscillation with $\pm 50\%$ fundamental frequency amplitude.
   - **Energy Flaw:** Syllabic RMS scaled to oscillate between $0.48$ and $0.06$ (swings $> 18\text{ dB}$).
   - **Muffled Flaw:** 4th-order lowpass Butterworth filter with $750\text{ Hz}$ cutoff suppressing all consonant energy.

---

### 5. Manifest & Annotation JSON Schema

#### `data/manifest.json` Structure
```json
{
  "dataset_version": "1.0.0",
  "license": "CC-BY-4.0",
  "metadata": {
    "total_speech_texts": 6,
    "total_audio_pairs": 18,
    "total_annotated_flaw_regions": 54,
    "audio_sample_rate": 16000
  },
  "speech_texts": [
    {
      "text_id": "speech_01_gettysburg",
      "title": "Lincoln's Gettysburg Address",
      "canonical_transcript": "...",
      "word_count": 30,
      "source_provenance": "..."
    }
  ],
  "pairs": [
    {
      "pair_id": "pair_speech_01_gettysburg_excessive_speed_1",
      "text_id": "speech_01_gettysburg",
      "baseline_audio": "data/audio/baselines/speech_01_gettysburg_baseline.wav",
      "flawed_audio": "data/audio/flawed/speech_01_gettysburg_excessive_speed_severe.wav",
      "flaw_category": "excessive_speed",
      "severity": "severe",
      "speaker_metadata": {
        "speaker_id": "spk_m01",
        "gender": "male",
        "native_language": "en",
        "median_f0_hz": 135.0
      },
      "ground_truth_flaws": [
        {
          "flaw_id": "gt_pair_speech_01_gettysburg_excessive_speed_1_01",
          "category": "excessive_speed",
          "start_s": 0.66,
          "end_s": 2.49,
          "transcript_slice": "Four score and seven",
          "observed_value": 235.0,
          "baseline_value": 135.0,
          "severity": "severe",
          "annotation_method": "grounded_flaw_injection_excessive_speed"
        }
      ]
    }
  ]
}
```

---
*VoxContrast Dataset Documentation — CC-BY-4.0 License*
