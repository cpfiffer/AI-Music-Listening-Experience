#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

try:
    import soundfile as sf

    HAS_SF = True
except Exception:
    HAS_SF = False
    from scipy.io import wavfile


SR_TARGET = 22050
HOP = 512
N_FFT = 2048
EPS = 1e-10
SUPPORTED_EXTS = {".wav", ".wave", ".mp3", ".flac", ".aif", ".aiff", ".m4a"}


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "clip"


def ensure_wav(path: Path):
    ext = path.suffix.lower()
    if ext in {".wav", ".wave"}:
        return path, lambda: None
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"ffmpeg required to inspect {ext} files, but ffmpeg is not installed")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:500])
    return tmp, lambda: tmp.unlink(missing_ok=True)


def load_audio_mono(path: Path):
    if HAS_SF:
        y, sr = sf.read(str(path), always_2d=True)
        y = y.mean(axis=1).astype(np.float32)
        return y, int(sr)
    sr, y = wavfile.read(str(path))
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    max_abs = float(np.max(np.abs(y)) + EPS)
    if max_abs > 1.5:
        y = y / max_abs
    return y.astype(np.float32), int(sr)


def resample_to_target(y, sr):
    if sr == SR_TARGET:
        return y, sr
    g = math.gcd(sr, SR_TARGET)
    up = SR_TARGET // g
    down = sr // g
    return signal.resample_poly(y, up, down).astype(np.float32), SR_TARGET


def stft_mag(y, sr):
    f, t, zxx = signal.stft(
        y,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    mag = np.abs(zxx).astype(np.float32)
    return f, t, mag


def frame_rms_from_mag(mag):
    power = np.mean(mag**2, axis=0)
    return np.sqrt(power + EPS)


def spectral_centroid(freqs, mag):
    num = np.sum(freqs[:, None] * mag, axis=0)
    den = np.sum(mag, axis=0) + EPS
    return num / den


def spectral_flux(mag):
    diff = np.diff(mag, axis=1)
    diff = np.maximum(diff, 0.0)
    flux = np.sum(diff, axis=0)
    return np.concatenate([[0.0], flux])


def spectral_flatness(mag):
    x = np.maximum(mag, EPS)
    geom = np.exp(np.mean(np.log(x), axis=0))
    arith = np.mean(x, axis=0) + EPS
    return geom / arith


def smooth_1d(x, win=5):
    if len(x) == 0 or win <= 1:
        return x
    win = min(int(win), len(x))
    if win <= 1:
        return x
    w = np.ones(win, dtype=float) / win
    return np.convolve(x, w, mode="same")


def normalize_0_1(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


def triangular_score(x, center, width):
    if width <= 0:
        return 0.0
    return clamp01(1.0 - abs(x - center) / width)


def zero_crossing_rate(y):
    if len(y) < 2:
        return 0.0
    signs = np.signbit(y).astype(np.int8)
    return float(np.mean(np.abs(np.diff(signs))))


def local_peaks(x, threshold):
    peaks = []
    if len(x) < 3:
        return peaks
    for i in range(1, len(x) - 1):
        if x[i] >= threshold and x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
    return peaks


def band_energy_ratios(freqs, mag):
    power = mag**2
    total = float(np.sum(power) + EPS)
    low = float(np.sum(power[(freqs >= 20) & (freqs < 250), :]))
    mid = float(np.sum(power[(freqs >= 250) & (freqs < 4000), :]))
    high = float(np.sum(power[(freqs >= 4000), :]))
    return {
        "low": low / total,
        "mid": mid / total,
        "high": high / total,
    }


def compute_descriptor_scores(meta):
    centroid = meta["spectral_centroid_mean_hz"]
    flatness = meta["spectral_flatness_mean"]
    zcr = meta["zero_crossing_rate"]
    duration = meta["duration_s"]
    onsets = meta["onsets_per_second"]
    front = meta["front_loadedness"]
    tail = meta["tail_decay_score"]
    start_abrupt = meta["start_abruptness"]
    end_abrupt = meta["end_abruptness"]
    crest = meta["crest_factor"]
    low = meta["spectral_balance"]["low"]
    mid = meta["spectral_balance"]["mid"]
    high = meta["spectral_balance"]["high"]

    bright = max(clamp01((centroid - 1800.0) / 2500.0), clamp01((high - 0.05) / 0.14))
    dark = max(clamp01((900.0 - centroid) / 900.0), clamp01((low - 0.55) / 0.30))
    sub = 0.6 * clamp01((low - 0.55) / 0.30) + 0.4 * clamp01((350.0 - centroid) / 350.0)
    noise = max(
        clamp01((flatness - 0.20) / 0.35),
        clamp01((zcr - 0.08) / 0.18),
        clamp01((high - 0.09) / 0.18),
    )
    tonal = min(
        1.0,
        0.5 * clamp01((0.35 - flatness) / 0.25)
        + 0.3 * clamp01((0.16 - zcr) / 0.12)
        + 0.2 * clamp01((mid - 0.35) / 0.35),
    )
    transient = max(
        clamp01((front - 0.22) / 0.32),
        clamp01((crest - 3.0) / 5.0),
        clamp01((start_abrupt - 0.90) / 1.20),
        clamp01((onsets - 2.40) / 2.0),
    )
    sustain = min(
        1.0,
        0.45 * clamp01((duration - 2.0) / 8.0)
        + 0.35 * clamp01((tail - 0.45) / 0.40)
        + 0.20 * clamp01((1.5 - onsets) / 1.5),
    )
    vocalish = min(
        1.0,
        0.25 * triangular_score(centroid, 1500.0, 1000.0)
        + 0.20 * clamp01((0.35 - flatness) / 0.25)
        + 0.15 * clamp01((0.18 - zcr) / 0.15)
        + 0.20 * clamp01((mid - 0.45) / 0.35)
        + 0.20 * (clamp01((duration - 0.35) / 0.75) * clamp01((8.0 - duration) / 7.0)),
    )
    soft_entry = clamp01((0.60 - start_abrupt) / 0.60)
    soft_tail = clamp01((0.50 - end_abrupt) / 0.50) * clamp01((tail - 0.50) / 0.50)
    hard_cut = clamp01((end_abrupt - 0.55) / 1.25) * clamp01((0.50 - tail) / 0.50)

    return {
        "bright": bright,
        "dark": dark,
        "sub": sub,
        "noise": noise,
        "tonal": tonal,
        "transient": transient,
        "sustain": sustain,
        "vocalish": vocalish,
        "soft_entry": soft_entry,
        "soft_tail": soft_tail,
        "hard_cut": hard_cut,
    }


DESCRIPTOR_KEYS = ["bright", "dark", "sub", "noise", "tonal", "transient", "sustain", "vocalish", "soft_entry", "soft_tail", "hard_cut"]


def descriptor_vector(scores):
    return np.array([scores[k] for k in DESCRIPTOR_KEYS], dtype=np.float64)


def descriptor_distance(vec_a, vec_b):
    return float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))


def compute_similarity_index(results, k_similar=5, k_contrast=5):
    """For each clip, find the k nearest and k furthest clips in descriptor space.
    
    For large datasets (>2000 clips), uses a chunked approach to avoid
    allocating an n*n distance matrix.
    """
    n = len(results)
    vectors = np.array([descriptor_vector(r["meta"]["descriptor_scores"]) for r in results])

    USE_FULL_MATRIX = n <= 2000

    if USE_FULL_MATRIX:
        # small dataset: full pairwise matrix
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = vectors - vectors[i]
            dist_matrix[i] = np.sqrt(np.sum(diff ** 2, axis=1))

        similarity_index = {}
        for i in range(n):
            dists = dist_matrix[i]
            order = np.argsort(dists)
            similar_indices = [j for j in order if j != i][:k_similar]
            contrast_indices = [j for j in reversed(order) if j != i][:k_contrast]
            rel_path = results[i]["relative_path"]
            similarity_index[rel_path] = {
                "similar": [{"relative_path": results[j]["relative_path"], "distance": round(float(dists[j]), 4), "index": int(j)} for j in similar_indices],
                "contrast": [{"relative_path": results[j]["relative_path"], "distance": round(float(dists[j]), 4), "index": int(j)} for j in contrast_indices],
            }
    else:
        # large dataset: compute per-row distances without full matrix
        k_need = max(k_similar, k_contrast)
        similarity_index = {}
        for i in range(n):
            if (i + 1) % 2000 == 0:
                print(f"  similarity: [{i + 1}/{n}]")
            diff = vectors - vectors[i]
            dists = np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)
            dists[i] = np.inf  # exclude self from similar
            sim_idx = np.argpartition(dists, k_similar)[:k_similar]
            sim_idx = sim_idx[np.argsort(dists[sim_idx])]

            dists[i] = -np.inf  # exclude self from contrast
            con_idx = np.argpartition(-dists, k_contrast)[:k_contrast]
            con_idx = con_idx[np.argsort(-dists[con_idx])]
            dists[i] = 0.0

            rel_path = results[i]["relative_path"]
            similarity_index[rel_path] = {
                "similar": [{"relative_path": results[j]["relative_path"], "distance": round(float(dists[j]), 4), "index": int(j)} for j in sim_idx],
                "contrast": [{"relative_path": results[j]["relative_path"], "distance": round(float(dists[j]), 4), "index": int(j)} for j in con_idx],
            }

    return similarity_index


def descriptor_tags(meta, scores):
    tags = []
    duration = meta["duration_s"]

    if scores["sub"] >= 0.58:
        tags.append("sub-heavy")
    if scores["bright"] >= 0.55:
        tags.append("bright")
    if scores["dark"] >= 0.55:
        tags.append("dark")
    if scores["noise"] >= 0.60:
        tags.append("noisy")
    if scores["tonal"] >= 0.58:
        tags.append("tonal")
    if scores["transient"] >= 0.60:
        tags.append("transient-heavy")
    if scores["sustain"] >= 0.58:
        tags.append("sustained")
    if scores["vocalish"] >= 0.60:
        tags.append("vocalish")
    if scores["soft_entry"] >= 0.58:
        tags.append("soft-entry")
    if scores["soft_tail"] >= 0.55:
        tags.append("soft-tail")
    if scores["hard_cut"] >= 0.45:
        tags.append("hard-cut")
    if duration <= 0.5 or (duration <= 1.2 and scores["transient"] >= 0.68):
        tags.append("one-shot")

    return tags


def analyze_clip(path: Path):
    wav_path, cleanup = ensure_wav(path)
    try:
        y, sr = load_audio_mono(wav_path)
        if len(y) == 0:
            raise RuntimeError("empty audio")
        y, sr = resample_to_target(y, sr)
        duration_s = len(y) / float(sr)

        freqs, t_frames, mag = stft_mag(y, sr)
        rms = frame_rms_from_mag(mag)
        centroid = spectral_centroid(freqs, mag)
        flux = spectral_flux(mag)
        onset = normalize_0_1(smooth_1d(flux, win=7))
        flatness = spectral_flatness(mag)
        balance = band_energy_ratios(freqs, mag)

        energy_total = float(np.sum(rms) + EPS)
        n = len(rms)
        first_n = max(1, int(round(n * 0.2)))
        last_n = max(1, int(round(n * 0.15)))
        front_loadedness = float(np.sum(rms[:first_n]) / energy_total)
        tail_energy_fraction = float(np.sum(rms[-last_n:]) / energy_total)
        peak_rms = float(np.max(rms) + EPS)
        ending_rms_to_peak = float(np.mean(rms[-last_n:]) / peak_rms)
        start_window = max(1, int(sr * 0.05))
        end_window = max(1, int(sr * 0.05))
        start_rms = float(np.sqrt(np.mean(y[:start_window] ** 2) + EPS))
        end_rms = float(np.sqrt(np.mean(y[-end_window:] ** 2) + EPS))
        global_wave_rms = float(np.sqrt(np.mean(y**2) + EPS))
        peak_amp = float(np.max(np.abs(y)))
        zcr = zero_crossing_rate(y)
        crest_factor = float(peak_amp / (global_wave_rms + EPS))

        onset_threshold = float(np.mean(onset) + np.std(onset))
        peaks = local_peaks(onset, onset_threshold)
        onsets_per_second = float(len(peaks) / max(duration_s, EPS))

        tail_decay_score = float(max(0.0, min(1.0, 1.0 - ending_rms_to_peak)))
        start_abruptness = float(start_rms / (global_wave_rms + EPS))
        end_abruptness = float(end_rms / (global_wave_rms + EPS))

        summary = {
            "filename": path.name,
            "path": str(path),
            "duration_s": float(duration_s),
            "sample_rate_hz": int(sr),
            "peak_amplitude": peak_amp,
            "mean_rms": float(np.mean(rms)),
            "peak_rms": float(np.max(rms)),
            "rms_std": float(np.std(rms)),
            "spectral_centroid_mean_hz": float(np.mean(centroid)),
            "spectral_centroid_std_hz": float(np.std(centroid)),
            "spectral_flux_mean": float(np.mean(flux)),
            "spectral_flux_std": float(np.std(flux)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "zero_crossing_rate": zcr,
            "crest_factor": crest_factor,
            "onsets_per_second": onsets_per_second,
            "front_loadedness": front_loadedness,
            "tail_energy_fraction": tail_energy_fraction,
            "tail_decay_score": tail_decay_score,
            "ending_rms_to_peak": ending_rms_to_peak,
            "start_abruptness": start_abruptness,
            "end_abruptness": end_abruptness,
            "spectral_balance": balance,
        }
        summary["descriptor_scores"] = compute_descriptor_scores(summary)
        summary["descriptor_tags"] = descriptor_tags(summary, summary["descriptor_scores"])
        summary["suggested_roles"] = suggest_roles(summary)

        arrays = {
            "waveform": y,
            "waveform_sr": sr,
            "t_frames": t_frames,
            "freqs": freqs,
            "mag": mag,
            "rms": rms,
            "centroid": centroid,
            "onset": onset,
        }
        return summary, arrays
    finally:
        cleanup()


def suggest_roles(meta):
    d = meta["duration_s"]
    onset_density = meta["onsets_per_second"]
    flatness = meta["spectral_flatness_mean"]
    centroid = meta["spectral_centroid_mean_hz"]
    front = meta["front_loadedness"]
    tail = meta["tail_decay_score"]
    end_ratio = meta["ending_rms_to_peak"]
    high = meta["spectral_balance"]["high"]
    tags = set(meta.get("descriptor_tags", []))
    scores = meta.get("descriptor_scores", {})
    is_one_shot = "one-shot" in tags

    roles = []

    if is_one_shot or d <= 0.95 or (onset_density >= 3.0 and d < 3.0):
        roles.append("fragment")

    if d >= 1.5 and onset_density <= 1.4 and not is_one_shot and (
        scores.get("sustain", 0.0) >= 0.45 or "noisy" in tags or "soft-tail" in tags or tail >= 0.45
    ):
        roles.append("texture")

    if 0.5 <= d <= 8.0 and onset_density <= 2.8 and not is_one_shot and (
        scores.get("vocalish", 0.0) >= 0.45 or "tonal" in tags or flatness <= 0.35
    ):
        roles.append("phrase")

    if tail >= 0.65 and front <= 0.40 and ("soft-entry" in tags or scores.get("sustain", 0.0) >= 0.45):
        roles.append("bloom")

    if (front >= 0.35 and d <= 4.0) or (is_one_shot and ("transient-heavy" in tags or "bright" in tags)):
        roles.append("transition")

    if "hard-cut" in tags or (
        end_ratio <= 0.2 and ("transient-heavy" in tags or "bright" in tags or "noisy" in tags or high >= 0.12 or centroid >= 1800)
    ):
        roles.append("resample-fodder")

    deduped = []
    for role in roles:
        if role not in deduped:
            deduped.append(role)

    if not deduped:
        deduped.append("texture" if d >= 2.0 else "fragment")
    return deduped


def plot_waveform(y, sr, out_path: Path, title: str):
    x = np.arange(len(y)) / float(sr)
    plt.figure(figsize=(12, 2.8))
    plt.plot(x, y, linewidth=0.7)
    plt.title(f"Waveform — {title}")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_spectrogram(freqs, t_frames, mag, out_path: Path, title: str):
    db = 20 * np.log10(np.maximum(mag, EPS))
    max_freq = 8000
    mask = freqs <= max_freq
    plt.figure(figsize=(12, 4))
    plt.imshow(
        db[mask, :],
        origin="lower",
        aspect="auto",
        extent=[float(t_frames[0]) if len(t_frames) else 0.0, float(t_frames[-1]) if len(t_frames) else 0.0, float(freqs[mask][0]) if np.any(mask) else 0.0, float(freqs[mask][-1]) if np.any(mask) else max_freq],
        cmap="magma",
    )
    plt.title(f"Spectrogram — {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_features(t_frames, rms, centroid, onset, out_path: Path, title: str):
    if len(t_frames) == 0:
        t_frames = np.array([0.0])
        rms = np.array([0.0])
        centroid = np.array([0.0])
        onset = np.array([0.0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 5.2), sharex=True)
    axes[0].plot(t_frames, normalize_0_1(rms), color="#2C7BE5")
    axes[0].set_ylabel("RMS")
    axes[0].set_title(f"Feature overview — {title}")

    axes[1].plot(t_frames, normalize_0_1(centroid), color="#E5532C")
    axes[1].set_ylabel("Bright")

    axes[2].plot(t_frames, normalize_0_1(onset), color="#2CB57B")
    axes[2].set_ylabel("Onset")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def discover_audio_files(input_dir: Path):
    files = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return sorted(files)


def format_float(x, ndigits=3):
    return f"{x:.{ndigits}f}"


def clip_group(relative_path: str) -> str:
    parts = relative_path.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def build_dashboard(results):
    clips = [r["meta"] for r in results]
    role_counts = {}
    tag_counts = {}
    groups = {}
    for result in results:
        meta = result["meta"]
        group = clip_group(result["relative_path"])
        groups[group] = groups.get(group, 0) + 1
        for role in meta["suggested_roles"]:
            role_counts[role] = role_counts.get(role, 0) + 1
        for tag in meta.get("descriptor_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    def stat(key):
        values = [c[key] for c in clips]
        return {
            "min": float(min(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    return {
        "role_counts": dict(sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[0])),
        "stats": {
            "duration_s": stat("duration_s"),
            "spectral_centroid_mean_hz": stat("spectral_centroid_mean_hz"),
            "onsets_per_second": stat("onsets_per_second"),
            "front_loadedness": stat("front_loadedness"),
            "tail_decay_score": stat("tail_decay_score"),
        },
        "extremes": {
            "brightest": max(results, key=lambda r: r["meta"]["spectral_centroid_mean_hz"]),
            "darkest": min(results, key=lambda r: r["meta"]["spectral_centroid_mean_hz"]),
            "most_front_loaded": max(results, key=lambda r: r["meta"]["front_loadedness"]),
            "strongest_tail_decay": max(results, key=lambda r: r["meta"]["tail_decay_score"]),
        },
    }


def candidate_set_definitions():
    return [
        {
            "name": "vocalish_phrases",
            "description": "Phrase-like clips with strong vocalish character",
            "predicate": lambda m: "phrase" in m["suggested_roles"] and m["descriptor_scores"]["vocalish"] >= 0.55,
            "score": lambda m: 0.6 * m["descriptor_scores"]["vocalish"] + 0.25 * m["tail_decay_score"] + 0.15 * m["descriptor_scores"]["tonal"],
        },
        {
            "name": "bloom_tails",
            "description": "Clips with strong trailing-decay behavior",
            "predicate": lambda m: "bloom" in m["suggested_roles"],
            "score": lambda m: 0.65 * m["tail_decay_score"] + 0.2 * m["descriptor_scores"]["soft_entry"] + 0.15 * m["descriptor_scores"]["sustain"],
        },
        {
            "name": "dark_textures",
            "description": "Dark or sub-heavy texture material",
            "predicate": lambda m: "texture" in m["suggested_roles"] and ("dark" in m.get("descriptor_tags", []) or "sub-heavy" in m.get("descriptor_tags", [])),
            "score": lambda m: 0.45 * m["descriptor_scores"]["dark"] + 0.35 * m["descriptor_scores"]["sub"] + 0.2 * m["descriptor_scores"]["sustain"],
        },
        {
            "name": "bright_transients",
            "description": "Bright, transient-heavy material for cuts and accents",
            "predicate": lambda m: "bright" in m.get("descriptor_tags", []) and "transient-heavy" in m.get("descriptor_tags", []),
            "score": lambda m: 0.45 * m["descriptor_scores"]["bright"] + 0.45 * m["descriptor_scores"]["transient"] + 0.10 * m["descriptor_scores"]["hard_cut"],
        },
        {
            "name": "one_shot_impacts",
            "description": "Short impact-style clips and one-shots",
            "predicate": lambda m: "one-shot" in m.get("descriptor_tags", []) or ("fragment" in m["suggested_roles"] and "transition" in m["suggested_roles"]),
            "score": lambda m: 0.5 * m["descriptor_scores"]["transient"] + 0.3 * m["front_loadedness"] + 0.2 * m["descriptor_scores"]["bright"],
        },
        {
            "name": "low_end_anchors",
            "description": "Low-end anchors, kicks, and sub-heavy elements",
            "predicate": lambda m: "sub-heavy" in m.get("descriptor_tags", []),
            "score": lambda m: 0.6 * m["descriptor_scores"]["sub"] + 0.2 * m["descriptor_scores"]["dark"] + 0.2 * m["descriptor_scores"]["sustain"],
        },
        {
            "name": "resample_fodder",
            "description": "Material likely to mutate well under chopping/resampling",
            "predicate": lambda m: "resample-fodder" in m["suggested_roles"],
            "score": lambda m: 0.45 * m["descriptor_scores"]["hard_cut"] + 0.30 * m["descriptor_scores"]["transient"] + 0.25 * m["descriptor_scores"]["bright"],
        },
    ]


def candidate_row(result, score):
    meta = result["meta"]
    return {
        "relative_path": result["relative_path"],
        "group": clip_group(result["relative_path"]),
        "roles": meta["suggested_roles"],
        "tags": meta.get("descriptor_tags", []),
        "score": float(score),
        "duration_s": meta["duration_s"],
        "spectral_centroid_mean_hz": meta["spectral_centroid_mean_hz"],
        "onsets_per_second": meta["onsets_per_second"],
        "front_loadedness": meta["front_loadedness"],
        "tail_decay_score": meta["tail_decay_score"],
        "source_path": meta["path"],
    }


def write_candidate_sets(results, out_dir: Path):
    candidate_dir = out_dir / "candidate_sets"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in candidate_set_definitions():
        selected = []
        for result in results:
            meta = result["meta"]
            if spec["predicate"](meta):
                score = spec["score"](meta)
                selected.append((result, score))

        selected.sort(key=lambda item: item[1], reverse=True)
        rows = [candidate_row(result, score) for result, score in selected]

        json_path = candidate_dir / f"{spec['name']}.json"
        csv_path = candidate_dir / f"{spec['name']}.csv"
        m3u_path = candidate_dir / f"{spec['name']}.m3u"

        json_path.write_text(
            json.dumps(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "count": len(rows),
                    "clips": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "relative_path",
                    "group",
                    "roles",
                    "tags",
                    "score",
                    "duration_s",
                    "spectral_centroid_mean_hz",
                    "onsets_per_second",
                    "front_loadedness",
                    "tail_decay_score",
                    "source_path",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "roles": ",".join(row["roles"]),
                        "tags": ",".join(row["tags"]),
                    }
                )

        m3u_lines = ["#EXTM3U"]
        for row in rows:
            m3u_lines.append(f"#EXTINF:-1,{row['relative_path']}")
            m3u_lines.append(row["source_path"])
        m3u_path.write_text("\n".join(m3u_lines) + "\n", encoding="utf-8")

        manifest.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "count": len(rows),
                "json": str(json_path.relative_to(out_dir)),
                "csv": str(csv_path.relative_to(out_dir)),
                "m3u": str(m3u_path.relative_to(out_dir)),
            }
        )

    manifest_path = out_dir / "candidate_sets.json"
    manifest_path.write_text(json.dumps({"sets": manifest}, indent=2), encoding="utf-8")
    return manifest


def write_csv(results, out_dir: Path):
    csv_path = out_dir / "clips_summary.csv"
    fieldnames = [
        "relative_path",
        "group",
        "roles",
        "tags",
        "duration_s",
        "spectral_centroid_mean_hz",
        "onsets_per_second",
        "front_loadedness",
        "tail_decay_score",
        "start_abruptness",
        "end_abruptness",
        "mean_rms",
        "peak_rms",
        "spectral_flatness_mean",
        "zero_crossing_rate",
        "crest_factor",
        "bright_score",
        "dark_score",
        "sub_score",
        "noise_score",
        "tonal_score",
        "transient_score",
        "sustain_score",
        "vocalish_score",
        "soft_entry_score",
        "soft_tail_score",
        "hard_cut_score",
        "spectral_balance_low",
        "spectral_balance_mid",
        "spectral_balance_high",
        "source_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            meta = result["meta"]
            writer.writerow(
                {
                    "relative_path": result["relative_path"],
                    "group": clip_group(result["relative_path"]),
                    "roles": ",".join(meta["suggested_roles"]),
                    "tags": ",".join(meta.get("descriptor_tags", [])),
                    "duration_s": meta["duration_s"],
                    "spectral_centroid_mean_hz": meta["spectral_centroid_mean_hz"],
                    "onsets_per_second": meta["onsets_per_second"],
                    "front_loadedness": meta["front_loadedness"],
                    "tail_decay_score": meta["tail_decay_score"],
                    "start_abruptness": meta["start_abruptness"],
                    "end_abruptness": meta["end_abruptness"],
                    "mean_rms": meta["mean_rms"],
                    "peak_rms": meta["peak_rms"],
                    "spectral_flatness_mean": meta["spectral_flatness_mean"],
                    "zero_crossing_rate": meta["zero_crossing_rate"],
                    "crest_factor": meta["crest_factor"],
                    "bright_score": meta["descriptor_scores"]["bright"],
                    "dark_score": meta["descriptor_scores"]["dark"],
                    "sub_score": meta["descriptor_scores"]["sub"],
                    "noise_score": meta["descriptor_scores"]["noise"],
                    "tonal_score": meta["descriptor_scores"]["tonal"],
                    "transient_score": meta["descriptor_scores"]["transient"],
                    "sustain_score": meta["descriptor_scores"]["sustain"],
                    "vocalish_score": meta["descriptor_scores"]["vocalish"],
                    "soft_entry_score": meta["descriptor_scores"]["soft_entry"],
                    "soft_tail_score": meta["descriptor_scores"]["soft_tail"],
                    "hard_cut_score": meta["descriptor_scores"]["hard_cut"],
                    "spectral_balance_low": meta["spectral_balance"]["low"],
                    "spectral_balance_mid": meta["spectral_balance"]["mid"],
                    "spectral_balance_high": meta["spectral_balance"]["high"],
                    "source_path": meta["path"],
                }
            )


def write_index(results, errors, input_dir: Path, out_dir: Path, candidate_sets, similarity_index):
    index_path = out_dir / "index.html"
    dashboard = build_dashboard(results)
    all_roles = sorted({role for r in results for role in r["meta"]["suggested_roles"]})
    all_tags = sorted({tag for r in results for tag in r["meta"].get("descriptor_tags", [])})
    all_groups = sorted({clip_group(r["relative_path"]) for r in results})
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append("<title>Clip Inspector Report</title>")
    lines.append(
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:32px;line-height:1.45;background:#111;color:#eee}"
        "a{color:#8ecbff} .clip{border:1px solid #333;padding:20px;margin:24px 0;border-radius:12px;background:#181818}"
        ".meta{color:#bbb} .roles span,.pill{display:inline-block;background:#2b2b2b;border:1px solid #444;padding:2px 8px;border-radius:999px;margin:0 6px 6px 0}"
        "img{max-width:100%;height:auto;border-radius:8px;border:1px solid #333;background:#000}"
        "code{background:#222;padding:2px 6px;border-radius:6px} ul{margin-top:6px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin:18px 0 28px 0}"
        ".panel{border:1px solid #333;background:#181818;border-radius:12px;padding:16px}"
        "table{width:100%;border-collapse:collapse;font-size:14px} th,td{border-bottom:1px solid #2a2a2a;padding:8px 10px;text-align:left} th{position:sticky;top:0;background:#181818}"
        "tbody tr:hover{background:#1d1d1d} input,select{background:#1a1a1a;color:#eee;border:1px solid #444;border-radius:8px;padding:8px 10px}"
        ".controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:16px 0 18px 0}"
        ".overview-wrap{max-height:420px;overflow:auto;border:1px solid #333;border-radius:12px;background:#181818}"
        ".small{font-size:13px;color:#bbb} .hidden{display:none} audio{width:100%;margin:10px 0 14px 0}"
        ".neighbors{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}"
        ".neighbor-panel{border:1px solid #333;background:#1a1a1a;border-radius:10px;padding:14px}"
        ".neighbor-panel h4{margin:0 0 10px 0;font-size:14px;color:#aaa}"
        ".neighbor-item{padding:6px 0;border-bottom:1px solid #222;font-size:13px}"
        ".neighbor-item:last-child{border-bottom:none}"
        ".neighbor-item a{color:#8ecbff;text-decoration:none} .neighbor-item a:hover{text-decoration:underline}"
        ".neighbor-dist{color:#666;font-size:12px;margin-left:6px}"
        ".neighbor-item audio{width:100%;height:28px;margin:4px 0 0 0}</style>"
    )
    lines.append("</head><body>")
    lines.append("<h1>Clip Inspector Report</h1>")
    lines.append(f"<p class='meta'>Generated {html.escape(dt.datetime.now().isoformat(timespec='seconds'))}</p>")
    lines.append(f"<p class='meta'>Input directory: <code>{html.escape(str(input_dir))}</code></p>")
    lines.append(f"<p class='meta'>Analyzed clips: {len(results)} | Errors: {len(errors)}</p>")
    lines.append(
        f"<p class='meta'>Exports: <a href='clips_summary.json'>JSON</a> | <a href='clips_summary.csv'>CSV</a> | <a href='candidate_sets.json'>candidate sets</a> | <a href='similarity_index.json'>similarity</a></p>"
    )

    lines.append("<div class='grid'>")
    lines.append("<div class='panel'><h3>Role counts</h3>")
    for role, count in dashboard["role_counts"].items():
        lines.append(f"<span class='pill'>{html.escape(role)}: {count}</span>")
    lines.append("</div>")
    lines.append("<div class='panel'><h3>Character tags</h3>")
    for tag, count in dashboard["tag_counts"].items():
        lines.append(f"<span class='pill'>{html.escape(tag)}: {count}</span>")
    lines.append("</div>")
    lines.append("<div class='panel'><h3>Groups</h3>")
    for group, count in dashboard["groups"].items():
        lines.append(f"<span class='pill'>{html.escape(group)}: {count}</span>")
    lines.append("</div>")
    lines.append("<div class='panel'><h3>Metric ranges</h3><ul>")
    for key, label in [
        ("duration_s", "duration"),
        ("spectral_centroid_mean_hz", "brightness"),
        ("onsets_per_second", "onsets/sec"),
        ("front_loadedness", "front-loadedness"),
        ("tail_decay_score", "tail decay"),
    ]:
        stats = dashboard["stats"][key]
        lines.append(
            f"<li>{label}: min {format_float(stats['min'], 2)} | median {format_float(stats['median'], 2)} | max {format_float(stats['max'], 2)}</li>"
        )
    lines.append("</ul></div>")
    lines.append("<div class='panel'><h3>Extremes</h3><ul>")
    extremes = dashboard["extremes"]
    lines.append(f"<li>brightest: <code>{html.escape(extremes['brightest']['relative_path'])}</code></li>")
    lines.append(f"<li>darkest: <code>{html.escape(extremes['darkest']['relative_path'])}</code></li>")
    lines.append(f"<li>most front-loaded: <code>{html.escape(extremes['most_front_loaded']['relative_path'])}</code></li>")
    lines.append(f"<li>strongest tail decay: <code>{html.escape(extremes['strongest_tail_decay']['relative_path'])}</code></li>")
    lines.append("</ul></div>")
    lines.append("</div>")

    lines.append("<h2>Candidate sets</h2><div class='grid'>")
    for item in candidate_sets:
        lines.append("<div class='panel'>")
        lines.append(f"<h3>{html.escape(item['name'])} <span class='small'>({item['count']})</span></h3>")
        lines.append(f"<p class='small'>{html.escape(item['description'])}</p>")
        lines.append(
            f"<p><a href='{html.escape(item['json'])}'>JSON</a> | <a href='{html.escape(item['csv'])}'>CSV</a> | <a href='{html.escape(item['m3u'])}'>M3U</a></p>"
        )
        lines.append("</div>")
    lines.append("</div>")

    lines.append("<h2>Overview</h2>")
    lines.append("<div class='controls'>")
    lines.append("<input id='searchBox' type='text' placeholder='Search path or role'>")
    lines.append("<select id='roleFilter'><option value=''>All roles</option>")
    for role in all_roles:
        lines.append(f"<option value='{html.escape(role)}'>{html.escape(role)}</option>")
    lines.append("</select>")
    lines.append("<select id='tagFilter'><option value=''>All tags</option>")
    for tag in all_tags:
        lines.append(f"<option value='{html.escape(tag)}'>{html.escape(tag)}</option>")
    lines.append("</select>")
    lines.append("<select id='groupFilter'><option value=''>All groups</option>")
    for group in all_groups:
        lines.append(f"<option value='{html.escape(group)}'>{html.escape(group)}</option>")
    lines.append("</select>")
    lines.append(
        "<select id='sortSelect'>"
        "<option value='path'>Sort: path</option>"
        "<option value='duration_desc'>Duration high-low</option>"
        "<option value='duration_asc'>Duration low-high</option>"
        "<option value='brightness_desc'>Brightness high-low</option>"
        "<option value='onsets_desc'>Onsets high-low</option>"
        "<option value='front_desc'>Front-loaded high-low</option>"
        "<option value='tail_desc'>Tail decay high-low</option>"
        "</select>"
    )
    lines.append("<span class='small' id='matchCount'></span>")
    lines.append("</div>")

    lines.append("<div class='overview-wrap'><table id='overviewTable'><thead><tr>")
    lines.append("<th>path</th><th>group</th><th>roles</th><th>tags</th><th>dur</th><th>bright</th><th>onsets</th><th>front</th><th>tail</th><th>jump</th>")
    lines.append("</tr></thead><tbody>")
    for idx, result in enumerate(results):
        meta = result["meta"]
        rel = result["relative_path"]
        group = clip_group(rel)
        roles = ", ".join(meta["suggested_roles"])
        tags = ", ".join(meta.get("descriptor_tags", []))
        row_id = f"clip-{idx}"
        lines.append(
            f"<tr class='overview-row' data-index='{idx}' data-path='{html.escape(rel.lower())}' data-group='{html.escape(group)}' data-roles='{html.escape(' '.join(meta['suggested_roles']))}' data-tags='{html.escape(' '.join(meta.get('descriptor_tags', [])))}' data-duration='{meta['duration_s']}' data-brightness='{meta['spectral_centroid_mean_hz']}' data-onsets='{meta['onsets_per_second']}' data-front='{meta['front_loadedness']}' data-tail='{meta['tail_decay_score']}'>"
            f"<td>{html.escape(rel)}</td>"
            f"<td>{html.escape(group)}</td>"
            f"<td>{html.escape(roles)}</td>"
            f"<td>{html.escape(tags)}</td>"
            f"<td>{format_float(meta['duration_s'], 2)}</td>"
            f"<td>{format_float(meta['spectral_centroid_mean_hz'], 0)}</td>"
            f"<td>{format_float(meta['onsets_per_second'], 2)}</td>"
            f"<td>{format_float(meta['front_loadedness'], 2)}</td>"
            f"<td>{format_float(meta['tail_decay_score'], 2)}</td>"
            f"<td><a href='#{row_id}'>jump</a></td></tr>"
        )
    lines.append("</tbody></table></div>")

    if errors:
        lines.append("<h2>Errors</h2><ul>")
        for item in errors:
            lines.append(
                f"<li><code>{html.escape(item['path'])}</code>: {html.escape(item['error'])}</li>"
            )
        lines.append("</ul>")

    for idx, result in enumerate(results):
        meta = result["meta"]
        rel = result["relative_path"]
        group = clip_group(rel)
        card_id = f"clip-{idx}"
        lines.append(
            f"<div class='clip clip-card' id='{card_id}' data-path='{html.escape(rel.lower())}' data-group='{html.escape(group)}' data-roles='{html.escape(' '.join(meta['suggested_roles']))}' data-tags='{html.escape(' '.join(meta.get('descriptor_tags', [])))}' data-duration='{meta['duration_s']}' data-brightness='{meta['spectral_centroid_mean_hz']}' data-onsets='{meta['onsets_per_second']}' data-front='{meta['front_loadedness']}' data-tail='{meta['tail_decay_score']}'>"
        )
        lines.append(f"<h2>{html.escape(rel)}</h2>")
        lines.append(
            f"<p class='meta'><a href='{html.escape(result['source_uri'])}'>open source audio</a> | "
            f"<a href='{html.escape(result['meta_rel'])}'>metadata json</a></p>"
        )
        lines.append(f"<p class='meta'>group: <code>{html.escape(group)}</code></p>")
        lines.append(f"<audio controls preload='none' src='{html.escape(result['source_uri'])}'></audio>")
        lines.append("<div class='roles'>")
        for role in meta["suggested_roles"]:
            lines.append(f"<span>{html.escape(role)}</span>")
        lines.append("</div>")
        if meta.get("descriptor_tags"):
            lines.append("<div class='roles'>")
            for tag in meta["descriptor_tags"]:
                lines.append(f"<span>{html.escape(tag)}</span>")
            lines.append("</div>")
        lines.append("<ul>")
        lines.append(f"<li>duration: {format_float(meta['duration_s'], 2)}s</li>")
        lines.append(f"<li>mean brightness: {format_float(meta['spectral_centroid_mean_hz'], 0)} Hz</li>")
        lines.append(f"<li>onsets/sec: {format_float(meta['onsets_per_second'], 2)}</li>")
        lines.append(f"<li>front-loadedness: {format_float(meta['front_loadedness'], 2)}</li>")
        lines.append(f"<li>tail decay score: {format_float(meta['tail_decay_score'], 2)}</li>")
        lines.append(f"<li>zero crossing rate: {format_float(meta['zero_crossing_rate'], 3)}</li>")
        lines.append(f"<li>crest factor: {format_float(meta['crest_factor'], 2)}</li>")
        lines.append(
            f"<li>spectral balance: low {format_float(meta['spectral_balance']['low'], 2)} / "
            f"mid {format_float(meta['spectral_balance']['mid'], 2)} / "
            f"high {format_float(meta['spectral_balance']['high'], 2)}</li>"
        )
        lines.append(
            f"<li>descriptor scores: vocalish {format_float(meta['descriptor_scores']['vocalish'], 2)} / transient {format_float(meta['descriptor_scores']['transient'], 2)} / tonal {format_float(meta['descriptor_scores']['tonal'], 2)} / noise {format_float(meta['descriptor_scores']['noise'], 2)}</li>"
        )
        lines.append("</ul>")
        lines.append(f"<img src='{html.escape(result['waveform_rel'])}' alt='waveform'>")
        lines.append("<br><br>")
        lines.append(f"<img src='{html.escape(result['spectrogram_rel'])}' alt='spectrogram'>")
        lines.append("<br><br>")
        lines.append(f"<img src='{html.escape(result['features_rel'])}' alt='feature overview'>")

        # similarity / contrast panels
        sim_data = similarity_index.get(rel, {})
        similar = sim_data.get("similar", [])
        contrast = sim_data.get("contrast", [])
        if similar or contrast:
            lines.append("<div class='neighbors'>")
            if similar:
                lines.append("<div class='neighbor-panel'><h4>Similar clips</h4>")
                for nb in similar:
                    nb_idx = nb["index"]
                    nb_rel = nb["relative_path"]
                    nb_dist = nb["distance"]
                    nb_src = results[nb_idx]["source_uri"] if nb_idx < len(results) else ""
                    lines.append(
                        f"<div class='neighbor-item'>"
                        f"<a href='#clip-{nb_idx}'>{html.escape(nb_rel)}</a>"
                        f"<span class='neighbor-dist'>d={nb_dist:.3f}</span>"
                    )
                    if nb_src:
                        lines.append(f"<audio controls preload='none' src='{html.escape(nb_src)}'></audio>")
                    lines.append("</div>")
                lines.append("</div>")
            if contrast:
                lines.append("<div class='neighbor-panel'><h4>Contrast clips (most different)</h4>")
                for nb in contrast:
                    nb_idx = nb["index"]
                    nb_rel = nb["relative_path"]
                    nb_dist = nb["distance"]
                    nb_src = results[nb_idx]["source_uri"] if nb_idx < len(results) else ""
                    lines.append(
                        f"<div class='neighbor-item'>"
                        f"<a href='#clip-{nb_idx}'>{html.escape(nb_rel)}</a>"
                        f"<span class='neighbor-dist'>d={nb_dist:.3f}</span>"
                    )
                    if nb_src:
                        lines.append(f"<audio controls preload='none' src='{html.escape(nb_src)}'></audio>")
                    lines.append("</div>")
                lines.append("</div>")
            lines.append("</div>")

        lines.append("</div>")

    lines.append(
        "<script>"
        "const searchBox=document.getElementById('searchBox');"
        "const roleFilter=document.getElementById('roleFilter');"
        "const tagFilter=document.getElementById('tagFilter');"
        "const groupFilter=document.getElementById('groupFilter');"
        "const sortSelect=document.getElementById('sortSelect');"
        "const tableBody=document.querySelector('#overviewTable tbody');"
        "const rows=Array.from(document.querySelectorAll('.overview-row'));"
        "const cards=Array.from(document.querySelectorAll('.clip-card'));"
        "const matchCount=document.getElementById('matchCount');"
        "function metric(row,key){return parseFloat(row.dataset[key]||'0');}"
        "function compareRows(a,b,mode){"
        " if(mode==='duration_desc') return metric(b,'duration')-metric(a,'duration');"
        " if(mode==='duration_asc') return metric(a,'duration')-metric(b,'duration');"
        " if(mode==='brightness_desc') return metric(b,'brightness')-metric(a,'brightness');"
        " if(mode==='onsets_desc') return metric(b,'onsets')-metric(a,'onsets');"
        " if(mode==='front_desc') return metric(b,'front')-metric(a,'front');"
        " if(mode==='tail_desc') return metric(b,'tail')-metric(a,'tail');"
        " return a.dataset.path.localeCompare(b.dataset.path);"
        "}"
        "function applyFilters(){"
        " const q=(searchBox.value||'').trim().toLowerCase();"
        " const role=roleFilter.value;"
        " const tag=tagFilter.value;"
        " const group=groupFilter.value;"
        " let visible=0;"
        " rows.sort((a,b)=>compareRows(a,b,sortSelect.value));"
        " rows.forEach(row=>tableBody.appendChild(row));"
        " rows.forEach(row=>{"
        "   const matchQ=!q || row.dataset.path.includes(q) || row.dataset.roles.includes(q) || row.dataset.tags.includes(q);"
        "   const matchRole=!role || row.dataset.roles.split(' ').includes(role);"
        "   const matchTag=!tag || row.dataset.tags.split(' ').includes(tag);"
        "   const matchGroup=!group || row.dataset.group===group;"
        "   const show=matchQ && matchRole && matchTag && matchGroup;"
        "   row.style.display=show?'':'none';"
        "   const card=document.getElementById('clip-'+row.dataset.index);"
        "   if(card) card.style.display=show?'':'none';"
        "   if(show) visible += 1;"
        " });"
        " matchCount.textContent=visible + ' clips shown';"
        "}"
        "[searchBox,roleFilter,tagFilter,groupFilter,sortSelect].forEach(el=>el.addEventListener('input',applyFilters));"
        "[roleFilter,tagFilter,groupFilter,sortSelect].forEach(el=>el.addEventListener('change',applyFilters));"
        "applyFilters();"
        "</script>"
    )
    lines.append("</body></html>")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Inspect a folder of audio clips and generate visual/structural reports.")
    ap.add_argument("--input_dir", required=True, help="Folder of clips to inspect")
    ap.add_argument("--out_dir", required=True, help="Folder to write report artifacts")
    ap.add_argument("--limit", type=int, default=None, help="Optional maximum number of clips to inspect")
    ap.add_argument("--metadata-only", action="store_true", help="Skip plot generation, output only JSON/CSV/similarity data")
    args = ap.parse_args()

    metadata_only = args.metadata_only
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_only:
        clips_out = out_dir / "clips"
        clips_out.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(input_dir)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No audio files found under {input_dir}")

    results = []
    errors = []
    total = len(files)

    for i, path in enumerate(files):
        rel = str(path.relative_to(input_dir))
        if metadata_only and (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{total}] {rel}")
        try:
            meta, arrays = analyze_clip(path)
            meta["relative_path"] = rel

            result_entry = {
                "relative_path": rel,
                "source_uri": path.resolve().as_uri(),
                "meta": meta,
            }

            if not metadata_only:
                slug = slugify(rel)
                clip_dir = (out_dir / "clips") / slug
                clip_dir.mkdir(parents=True, exist_ok=True)

                waveform_path = clip_dir / "waveform.png"
                spectrogram_path = clip_dir / "spectrogram.png"
                features_path = clip_dir / "features.png"
                meta_path = clip_dir / "metadata.json"

                plot_waveform(arrays["waveform"], arrays["waveform_sr"], waveform_path, path.name)
                plot_spectrogram(arrays["freqs"], arrays["t_frames"], arrays["mag"], spectrogram_path, path.name)
                plot_features(arrays["t_frames"], arrays["rms"], arrays["centroid"], arrays["onset"], features_path, path.name)

                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                result_entry["meta_rel"] = str(meta_path.relative_to(out_dir))
                result_entry["waveform_rel"] = str(waveform_path.relative_to(out_dir))
                result_entry["spectrogram_rel"] = str(spectrogram_path.relative_to(out_dir))
                result_entry["features_rel"] = str(features_path.relative_to(out_dir))

            results.append(result_entry)
        except Exception as e:
            errors.append({"path": str(path), "error": str(e)})

    results.sort(key=lambda x: x["relative_path"])

    # compute similarity index
    similarity_index = compute_similarity_index(results)
    (out_dir / "similarity_index.json").write_text(json.dumps(similarity_index, indent=2), encoding="utf-8")

    aggregate = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "analyzed": len(results),
        "errors": errors,
        "dashboard": build_dashboard(results),
        "clips": [r["meta"] for r in results],
    }
    (out_dir / "clips_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_csv(results, out_dir)
    candidate_sets = write_candidate_sets(results, out_dir)
    aggregate["candidate_sets"] = candidate_sets
    aggregate["similarity_index"] = similarity_index
    (out_dir / "clips_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    if not metadata_only:
        write_index(results, errors, input_dir, out_dir, candidate_sets, similarity_index)
        print(f"Analyzed {len(results)} clips. Report: {out_dir / 'index.html'}")
    else:
        print(f"Analyzed {len(results)} clips (metadata-only). Output: {out_dir}")
    if errors:
        print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
