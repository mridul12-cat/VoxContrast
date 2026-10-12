# VoxContrast: Contrastive Speech Dataset Documentation
## Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

---

### 1. Dataset Overview & Provenance Architecture

The VoxContrast Dataset is an open-access, paired multimodal speech evaluation corpus designed for high-resolution temporal speech flaw grounding and contrastive delivery analysis.

> [!IMPORTANT]
> **AUTHORITATIVE SPOKEN SPEECH CORPUS (`data/audio/`, `data/manifest.json`):**
> The dataset consists of 24 verified spoken speech recordings (6 pristine baselines and 18 flawed contrastive counter-utterances) covering 6 canonical rhetorical transcripts.
> - **Phonetic Intelligibility:** Renders full human phonetic articulation, natural formant contours, and connected speech cadence without synthetic humming or programmatic glottal simplifications.
> - **Exact Ground Truth:** 18 confirmed temporal flaw interval annotations grounded directly in measured physical acoustic anomalies (`[start_s, end_s]`), recorded across `data/manifest.json` and `data/annotations/`.
> - **Split Partitioning:**
>   - `train_dev` split (4 speeches, 12 pairs): Speeches 01 to 04 (`Gettysburg`, `Churchill`, `Kennedy`, `Keynote`).
>   - `final_test` split (2 speeches, 6 pairs): Speeches 05 and 06 (`Defense`, `Debate`).
> - **Audio Format:** 24,000 Hz, 16-bit PCM, Mono WAV across all recordings.
> - **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`) & Public Domain texts.

---

### 2. Rhetorical Speech Texts

The 6 canonical speech texts span political oratory, perorations, commencement addresses, technology keynotes, scientific doctoral defenses, and competitive policy debate openings. The archives cited below supplied **text transcripts only**:

| Text ID | Title | Word Count | Rhetorical Style / Cadence | Text Transcript Provenance (Text Only) |
|---|---|---|---|---|
| `speech_01_gettysburg` | Lincoln's Gettysburg Address | 30 | Rhetorical solemnity, measured cadence, deliberate emphasis (135 WPM) | Abraham Lincoln (1863), Public Domain US Historical Archives |
| `speech_02_churchill` | Churchill's We Shall Fight on the Beaches | 33 | Perorational crescendo, heavy pause structure, defiant timbre (125 WPM) | Winston Churchill (1940), Public Domain Parliamentary Address |
| `speech_03_kennedy` | Kennedy's Inaugural Address | 39 | Antithetical balance, rhythmic cadence, presidential oratory (138 WPM) | John F. Kennedy (1961), Public Domain Presidential Inaugural |
| `speech_04_keynote` | Next-Gen AI Keynote Announcement | 25 | Corporate technology keynote, rapid forward momentum (140 WPM) | Open Speech Corpus (2024), CC-BY-4.0 Tech Speech Benchmark |
| `speech_05_defense` | Doctoral Dissertation Defense | 24 | Rapid academic exposition, polysyllabic density, technical clarity (145 WPM) | Academic Open Speech Repository (2023), CC-BY-4.0 Scientific Corpus |
| `speech_06_debate` | Policy Debate Opening | 29 | High-density structured argumentation, rapid staccato cadence (155 WPM) | National Debate Archive (2022), CC-BY-4.0 Public Speech Dataset |

---

### 3. Contrastive Pair & Flaw Matrix

Each speech text is paired with 3 distinct flawed versions across controlled acoustic anomaly categories and severity tiers, recorded in `data/manifest.json`:

| Pair ID | Speech Text | Flaw Category | Severity | Ground Truth Regions | Flawed Audio File |
|---|---|---|---|:---:|---|
| `pair_speech_01_gettysburg_excessive_speed_1` | Gettysburg | `excessive_speed` | severe | 1 | `speech_01_gettysburg_excessive_speed_severe.wav` |
| `pair_speech_01_gettysburg_unnatural_pause_2` | Gettysburg | `unnatural_pause` | severe | 1 | `speech_01_gettysburg_unnatural_pause_severe.wav` |
| `pair_speech_01_gettysburg_monotone_pitch_3` | Gettysburg | `monotone_pitch` | severe | 1 | `speech_01_gettysburg_monotone_pitch_severe.wav` |
| `pair_speech_02_churchill_unnatural_pause_1` | Churchill | `unnatural_pause` | severe | 1 | `speech_02_churchill_unnatural_pause_severe.wav` |
| `pair_speech_02_churchill_excessive_speed_2` | Churchill | `excessive_speed` | severe | 1 | `speech_02_churchill_excessive_speed_severe.wav` |
| `pair_speech_02_churchill_muffled_articulation_3` | Churchill | `muffled_articulation` | severe | 1 | `speech_02_churchill_muffled_articulation_severe.wav` |
| `pair_speech_03_kennedy_muffled_articulation_1` | Kennedy | `muffled_articulation` | severe | 1 | `speech_03_kennedy_muffled_articulation_severe.wav` |
| `pair_speech_03_kennedy_monotone_pitch_2` | Kennedy | `monotone_pitch` | severe | 1 | `speech_03_kennedy_monotone_pitch_severe.wav` |
| `pair_speech_03_kennedy_unstable_energy_3` | Kennedy | `unstable_energy` | severe | 1 | `speech_03_kennedy_unstable_energy_severe.wav` |
| `pair_speech_04_keynote_monotone_pitch_1` | Keynote | `monotone_pitch` | severe | 1 | `speech_04_keynote_monotone_pitch_severe.wav` |
| `pair_speech_04_keynote_excessive_speed_2` | Keynote | `excessive_speed` | severe | 1 | `speech_04_keynote_excessive_speed_severe.wav` |
| `pair_speech_04_keynote_monotone_energy_3` | Keynote | `monotone_energy` | moderate | 1 | `speech_04_keynote_monotone_energy_moderate.wav` |
| `pair_speech_05_defense_unstable_energy_1` | Defense | `unstable_energy` | severe | 1 | `speech_05_defense_unstable_energy_severe.wav` |
| `pair_speech_05_defense_unnatural_pause_2` | Defense | `unnatural_pause` | severe | 1 | `speech_05_defense_unnatural_pause_severe.wav` |
| `pair_speech_05_defense_muffled_articulation_3` | Defense | `muffled_articulation` | severe | 1 | `speech_05_defense_muffled_articulation_severe.wav` |
| `pair_speech_06_debate_monotone_energy_1` | Debate | `monotone_energy` | moderate | 1 | `speech_06_debate_monotone_energy_moderate.wav` |
| `pair_speech_06_debate_excessive_speed_2` | Debate | `excessive_speed` | severe | 1 | `speech_06_debate_excessive_speed_severe.wav` |
| `pair_speech_06_debate_monotone_pitch_3` | Debate | `monotone_pitch` | severe | 1 | `speech_06_debate_monotone_pitch_severe.wav` |

---

### 4. Manifest & Annotation JSON Schema

#### `data/manifest.json` Structure
```json
{
  "dataset_name": "VoxContrast Spoken Speech Dataset",
  "dataset_version": "2.0.0",
  "provenance_class": "spoken_speech_v2",
  "license": "CC-BY-4.0",
  "metadata": {
    "total_speech_texts": 6,
    "total_audio_utterances": 24,
    "total_audio_pairs": 18,
    "total_annotated_flaw_regions": 18,
    "audio_sample_rate": 24000,
    "audio_format": "PCM_16 mono WAV"
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
