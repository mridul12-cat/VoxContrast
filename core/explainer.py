from __future__ import annotations

from typing import Literal


def format_timestamp(sec: float) -> str:
    """Formats seconds into MM:SS.SS representation."""
    mins = int(sec // 60)
    secs = sec % 60
    return f"{mins:02d}:{secs:05.2f}"


def generate_flaw_explanation(
    category: str,
    start_s: float,
    end_s: float,
    observed_val: float,
    baseline_val: float,
    threshold_val: float,
    unit: str,
    severity: Literal["minor", "moderate", "severe"]
) -> tuple[str, str]:
    """
    Returns (explanation, caveat).
    Explanation includes exact observed value, baseline value, threshold value, and delta.
    Caveat provides scientific grounding separating measurement from subjective inference.
    """
    t_start_str = format_timestamp(start_s)
    t_end_str = format_timestamp(end_s)
    delta = observed_val - threshold_val
    pct_delta = abs(delta / threshold_val * 100.0) if threshold_val != 0 else 0.0

    if category == "excessive_speed":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, the estimated local speech rate reached "
            f"{observed_val:.1f} {unit} compared to a baseline of {baseline_val:.1f} {unit}, "
            f"exceeding the calibrated maximum threshold of {threshold_val:.1f} {unit} by "
            f"+{delta:.1f} {unit} (+{pct_delta:.1f}%). This accelerated articulation rate "
            f"compresses syllable durations, reducing listener phonetic intake."
        )
        caveat = (
            "The acoustic measurement reliably confirms rapid syllable timing; it does not "
            "independently prove speaker anxiety or lack of preparation."
        )

    elif category == "slow_speech":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, the local speech rate dropped to "
            f"{observed_val:.1f} {unit} against a baseline of {baseline_val:.1f} {unit}, "
            f"falling below the lower pacing threshold of {threshold_val:.1f} {unit} by "
            f"{abs(delta):.1f} {unit} (-{pct_delta:.1f}%). Extended vowel elongation slows "
            f"conversational momentum."
        )
        caveat = (
            "The acoustic measurement confirms prolonged syllable durations; it does not "
            "distinguish between deliberate solemn pacing and cognitive hesitation."
        )

    elif category == "unnatural_pause":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, an unvoiced silence interval of "
            f"{observed_val:.2f} {unit} was measured, exceeding the standard phrase boundary "
            f"threshold of {threshold_val:.2f} {unit} by +{delta:.2f} {unit} (+{pct_delta:.1f}%). "
            f"Standard speech transitions utilize breathing pauses between {baseline_val:.2f}s and 1.20s."
        )
        caveat = (
            "The acoustic measurement verifies a prolonged absence of voiced vocal energy; "
            "it cannot definitively classify whether the silence was a strategic rhetorical pause "
            "or an unintended memory lapse."
        )

    elif category == "monotone_energy":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, the voiced RMS energy dynamic standard "
            f"deviation was only {observed_val:.2f} {unit} versus a natural dynamic baseline of "
            f"{baseline_val:.2f} {unit}, falling below the expressive threshold of "
            f"{threshold_val:.2f} {unit} by -{abs(delta):.2f} {unit} (-{pct_delta:.1f}%). "
            f"Flat energy contours diminish perceived vocal emphasis."
        )
        caveat = (
            "The acoustic measurement indicates restricted vocal amplitude variation; "
            "it does not capture stylistic deadpan humor or calm instructional intent."
        )

    elif category == "unstable_energy":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, peak-to-peak energy fluctuation reached "
            f"{observed_val:.1f} {unit} against a controlled baseline of {baseline_val:.1f} {unit}, "
            f"exceeding the stability threshold of {threshold_val:.1f} {unit} by "
            f"+{delta:.1f} {unit} (+{pct_delta:.1f}%). Abrupt volume shifts create perceptual jarring."
        )
        caveat = (
            "The acoustic measurement detects erratic RMS amplitude variance; it does not "
            "discriminate between microphone distance shifts and intentional dramatic shout/whisper dynamics."
        )

    elif category == "monotone_pitch":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, speaker-normalized fundamental frequency "
            f"exhibited a standard deviation of only {observed_val:.2f} {unit} relative to a natural "
            f"baseline of {baseline_val:.2f} {unit}, falling below the inflection threshold of "
            f"{threshold_val:.2f} {unit} by -{abs(delta):.2f} {unit} (-{pct_delta:.1f}%). "
            f"A flat pitch contour produces perceptual monotony."
        )
        caveat = (
            "The acoustic measurement demonstrates narrow F0 variance in semitones; "
            "it does not infer speaker disinterest or emotional indifference."
        )

    elif category == "pitch_irregularity":
        explanation = (
            f"Between {t_start_str} and {t_end_str}, pitch variability spiked to "
            f"{observed_val:.2f} {unit} against a reference baseline of {baseline_val:.2f} {unit}, "
            f"surpassing the stability threshold of {threshold_val:.2f} {unit} by "
            f"+{delta:.2f} {unit} (+{pct_delta:.1f}%). Rapid unmodulated pitch breaks disrupt tonal coherence."
        )
        caveat = (
            "The acoustic measurement captures high semitone dispersion; it does not "
            "distinguish between vocal fatigue, pitch cracking, and stylized expressive intonation."
        )

    elif category in ("vocal_clarity", "muffled_articulation"):
        explanation = (
            f"Between {t_start_str} and {t_end_str}, the spectral centroid averaged "
            f"{observed_val:.0f} {unit} compared to a baseline of {baseline_val:.0f} {unit}, "
            f"dropping below the clarity threshold of {threshold_val:.0f} {unit} by "
            f"-{abs(delta):.0f} {unit} (-{pct_delta:.1f}%). Reduced high-frequency harmonic energy "
            f"indicates muffled phonation or reduced consonantal burst."
        )
        caveat = (
            "The acoustic measurement tracks frequency distribution of spectral energy; "
            "it cannot differentiate between room reverberation/low-pass filtering and speaker mumbling."
        )

    else:
        explanation = (
            f"Between {t_start_str} and {t_end_str}, measured {category} was {observed_val:.2f} {unit} "
            f"relative to baseline {baseline_val:.2f} {unit} and threshold {threshold_val:.2f} {unit}."
        )
        caveat = "Acoustic metric tracks physical signal variation, not subjective speaker intent."

    return explanation, caveat
