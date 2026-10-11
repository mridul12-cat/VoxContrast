from __future__ import annotations

import numpy as np
from core.schemas import TimeSeriesData


class SpeakerNormalizer:
    """Normalizes acoustic contours across speakers of differing pitch and volume baselines."""

    @staticmethod
    def compute_speaker_baseline(
        time_series: TimeSeriesData,
        silence_thresh_db: float = -38.0
    ) -> dict[str, float]:
        """Calculates speaker-specific baseline statistics from voiced frames."""
        voiced_f0 = [f for f in time_series.f0_contour if f > 0]
        voiced_rms = [db for db in time_series.rms_energy_db if db > silence_thresh_db]

        median_f0 = float(np.median(voiced_f0)) if voiced_f0 else 150.0
        mean_f0 = float(np.mean(voiced_f0)) if voiced_f0 else 150.0
        std_f0 = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0

        p90_energy = float(np.percentile(voiced_rms, 90)) if voiced_rms else -20.0
        mean_energy = float(np.mean(voiced_rms)) if voiced_rms else -25.0
        std_energy = float(np.std(voiced_rms)) if len(voiced_rms) > 1 else 3.0

        return {
            "median_f0_hz": round(median_f0, 2),
            "mean_f0_hz": round(mean_f0, 2),
            "std_f0_hz": round(std_f0, 2),
            "p90_energy_db": round(p90_energy, 2),
            "mean_energy_db": round(mean_energy, 2),
            "std_energy_db": round(std_energy, 2),
        }

    @staticmethod
    def normalize_f0_to_semitones(
        f0_contour: list[float],
        median_f0: float
    ) -> list[float]:
        """Converts Hertz fundamental frequencies to semitones relative to median F0."""
        if median_f0 <= 0:
            median_f0 = 150.0

        semitones = []
        for f in f0_contour:
            if f > 0:
                st = 12.0 * np.log2(f / median_f0)
                semitones.append(round(float(st), 2))
            else:
                semitones.append(0.0)
        return semitones

    @staticmethod
    def normalize_energy_relative(
        rms_energy_db: list[float],
        p90_energy: float
    ) -> list[float]:
        """Normalizes frame energy relative to 90th percentile voiced speech RMS."""
        norm_energy = [round(db - p90_energy, 2) for db in rms_energy_db]
        return norm_energy
