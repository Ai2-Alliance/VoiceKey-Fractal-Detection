#!/usr/bin/env python3
"""
VoiceKey Voice Authenticity Analyzer
====================================

Single evolved analyzer for the VoiceKey-Fractal-Detection project. Two
negative-detection screens over one recording:

1. MARGINAL COMPLEXITY SCREEN — sliding-window Higuchi Fractal Dimension
   (HFD) and Detrended Fluctuation Analysis (DFA) over the raw waveform,
   with sample-adaptive thresholds and per-window classification. This is
   the original fractal POC (2024), folded in.
2. NARRATIVE STRESS-COHERENCE SCREEN — does the speaker's physiological
   perturbation profile (jitter, shimmer, pitch, tempo) track the semantic
   stress profile of the specific challenge narrative they were given?

The marginal screen measures the distribution of signal complexity; the
coherence screen measures the CONDITIONAL signal — perturbation coupled to
content. Run both; they reject different failure modes.

Position in the VoiceKey pipeline
---------------------------------
This module is NEGATIVE DETECTION (see VoiceKey docs/negative_detection.md)
and runs ONLY after all positive-detection identity mechanics have passed:
biometric voice match, MFA, liveness, session binding. Positive detection
answers "is this the enrolled identity?"; this layer answers "is any
signature of synthesis present?" — the ABSENCE of human analog signatures
is the rejection signal. It contributes nothing to identification and must
never be used as a standalone authenticator.

Thesis: computational exhaustion vs pure analog
-----------------------------------------------
The claim is NOT that these signatures cannot be synthesized. A resourced
generator driving an emotion-controllable vocoder from an LLM arousal
annotation of the challenge text can pass a shallow version of the
coherence check; that bypass is acknowledged and priced in. The claim is
asymmetric cost, in the spirit of proof-of-work:

- A human produces both signals for free. They are analog side effects of
  a body phonating and a psyche processing content in real time — zero
  compute, zero preparation, at any depth and any duration.
- A generator must simulate them explicitly: match the marginal complexity
  statistics, estimate the arousal profile, render perturbations onto it,
  and keep every proxy (pitch, perturbation, tempo, pausing, and their
  co-movement) mutually consistent for the full sample, one-shot, in
  real time.

Attack cost therefore scales with challenge DEPTH — segment count, sample
duration, number of proxies, annotation subtlety, cross-proxy consistency —
while verification cost and the legitimate speaker's cost stay flat. Depth
is the security knob, tuned to the value of the unlock it gates, exactly as
work factors are tuned in password hashing: a low-value unlock warrants a
short shallow narrative; a high-value unlock warrants long, deep,
multi-axis coherence, at which point the generator is doing substantial
real-time modeling work to imitate what the analog system emits as exhaust.

Pipeline
--------
1. Load mono .wav (first N seconds, default 60), trim edge silence.
2. Marginal screen: sliding 1 s and 3 s windows over the waveform;
   HFD + DFA per window; sample-adaptive thresholds; per-window and
   overall classification; optional CSV/PNG export.
3. If a narrative annotation (JSON) is supplied — ordered segments, each
   with the text read aloud and an expected_arousal value in [0, 1]:
   a. Map segments onto the audio timeline proportionally by word count
      (or exactly, if an optional word-timestamp file from any ASR is
      given).
   b. Per segment, extract physiological proxies:
          - F0 mean (autocorrelation pitch, 75-400 Hz)
          - jitter proxy (frame-to-frame F0 perturbation, voiced runs)
          - shimmer proxy (frame-to-frame amplitude perturbation,
            voiced runs)
          - articulation rate proxy (envelope modulation spectrum)
          - pause ratio
          - HFD and DFA of the segment's energy envelope
   c. Build an observed-arousal composite per segment (z-scored jitter +
      shimmer + F0 mean + rate) and compute Spearman rank correlation
      against the narrative's expected-arousal profile, with a
      permutation p-value, plus per-feature correlations.
4. Report marginal statistics AND the coherence verdict.

Usage:
    python coherence_analyze.py audio.wav                   # marginal only
    python coherence_analyze.py audio.wav narrative.json    # both screens
    python coherence_analyze.py --selftest
Optional:
    --timestamps words.json  (list of {"word":..., "start":..., "end":...})
    --seconds 60             (analysis window length)
    --step 0.1               (marginal screen hop, seconds)
    --csv out.csv            (write per-window marginal results)
    --plot out.png           (write marginal plot; requires matplotlib)

Dependencies: numpy, scipy. matplotlib only if --plot is used.

Status: EXPERIMENTAL. Proof-of-mechanism: it exists to show both signals
are measurable, not to ship a production detector. Thresholds are
illustrative and require calibration on a real human corpus before any
operational use. Reader instructions for coherence challenges must permit
natural affect: the original test narrative's "maintain a consistent tone"
instruction suppresses exactly the signal the coherence screen measures,
and would manufacture false rejections there (it remains fine for the
marginal screen). MIT license, matching the parent project.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import spearmanr, zscore


# ----------------------------------------------------------------------
# Basic signal utilities
# ----------------------------------------------------------------------

def load_audio(path, seconds=60.0):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    peak = np.max(np.abs(x)) or 1.0
    x = x / peak
    n = int(seconds * sr)
    x = x[:n]
    # trim leading/trailing silence (2% of peak RMS, 20 ms frames)
    frame = max(1, int(0.02 * sr))
    rms = np.sqrt(np.convolve(x ** 2, np.ones(frame) / frame, mode="same"))
    thresh = 0.02 * np.max(rms)
    idx = np.where(rms > thresh)[0]
    if len(idx) > frame:
        x = x[idx[0]: idx[-1] + 1]
    return sr, x


def frame_signal(x, sr, win_s=0.030, hop_s=0.010):
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n_frames = max(0, 1 + (len(x) - win) // hop)
    if n_frames == 0:
        return np.empty((0, win)), hop
    frames = np.stack([x[i * hop: i * hop + win] for i in range(n_frames)])
    return frames, hop


def f0_autocorr(frames, sr, fmin=75.0, fmax=400.0):
    """Per-frame F0 via autocorrelation; NaN for unvoiced frames."""
    win = frames.shape[1]
    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), win - 1)
    f0 = np.full(len(frames), np.nan)
    energies = np.sqrt(np.mean(frames ** 2, axis=1))
    nonzero = energies[energies > 0]
    e_gate = 0.05 * (float(np.median(nonzero)) if len(nonzero) else 1.0)
    hann = np.hanning(win)
    for i, fr in enumerate(frames):
        if energies[i] < e_gate:
            continue
        fw = (fr - fr.mean()) * hann
        ac = np.correlate(fw, fw, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lag_min:lag_max]
        if len(seg) < 3:
            continue
        peaks, props = find_peaks(seg, height=0.30)
        if len(peaks) == 0:
            continue
        pk = peaks[np.argmax(props["peak_heights"])]
        lag = float(lag_min + pk)
        # parabolic interpolation for sub-sample lag precision, so
        # frame-to-frame F0 perturbation is measurable rather than
        # quantized away
        if 0 < pk < len(seg) - 1:
            y0, y1, y2 = seg[pk - 1], seg[pk], seg[pk + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-12:
                lag += 0.5 * (y0 - y2) / denom
        f0[i] = sr / lag
    return f0, energies


def _runs(mask):
    """Contiguous True runs in a boolean mask, as (start, end) pairs."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def jitter(f0):
    """Frame-to-frame F0 perturbation proxy (10 ms hop).

    Differences are taken only within contiguous voiced runs, so pause
    boundaries do not inject spurious perturbation. This is NOT Praat
    cycle-to-cycle jitter; treat it as a relative modulation proxy."""
    voiced = ~np.isnan(f0)
    vals = f0[voiced]
    if len(vals) < 4:
        return np.nan
    diffs = [np.abs(np.diff(f0[a:b])) for a, b in _runs(voiced) if b - a >= 2]
    if not diffs:
        return np.nan
    d = np.concatenate(diffs)
    if not len(d):
        return np.nan
    return float(np.mean(d) / np.mean(vals))


def shimmer(energies, voiced_mask):
    """Frame-to-frame amplitude perturbation proxy.

    Same voiced-run restriction as jitter(), for the same reason."""
    vals = energies[voiced_mask]
    if len(vals) < 4:
        return np.nan
    diffs = [np.abs(np.diff(energies[a:b]))
             for a, b in _runs(voiced_mask) if b - a >= 2]
    if not diffs:
        return np.nan
    d = np.concatenate(diffs)
    if not len(d):
        return np.nan
    return float(np.mean(d) / (np.mean(vals) or 1.0))


def onset_rate(x, sr):
    """Articulation-rate proxy: dominant syllabic modulation frequency.

    Uses the envelope modulation spectrum (1.5-9 Hz band) rather than
    peak counting; peak counting double-counts wide syllable humps under
    amplitude noise, on synthetic and real speech alike."""
    if len(x) < int(0.5 * sr):
        return np.nan
    env = np.abs(x)
    win = int(0.050 * sr)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    if np.max(env) == 0:
        return np.nan
    env = env - np.mean(env)
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / sr)
    band = (freqs >= 1.5) & (freqs <= 9.0)
    if not band.any() or np.max(spec[band]) == 0:
        return np.nan
    return float(freqs[band][np.argmax(spec[band])])


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension, standard normalization.

    Comparability caveat: the legacy analyze.py (2024 POC, since folded
    into this module) omitted the final /k in the curve-length
    normalization, which shifts its slope by ~1 relative to the standard
    definition used here. HFD values in the 2024 published CSVs are NOT
    comparable to this module's output — only to each other."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < kmax * 4:
        return np.nan
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            d = np.sum(np.abs(np.diff(x[idx])))
            Lk.append(d * (N - 1) / (len(idx) * k * k))
        if Lk:
            L.append(np.mean(Lk))
    L = np.array(L)
    if np.any(L <= 0):
        return np.nan
    k_arr = np.arange(1, len(L) + 1)
    slope = np.polyfit(np.log(1.0 / k_arr), np.log(L), 1)[0]
    return float(slope)


def dfa(x, scales=None):
    """Detrended Fluctuation Analysis exponent.

    Linear detrending is vectorized per scale (least-squares via a
    precomputed design matrix) so the sliding marginal screen stays
    tractable on full-rate waveform windows. Scale selection (12
    log-spaced scales, 8 .. N/4) also differs from the legacy POC's
    dense 5..100 range; DFA values are likewise not comparable to the
    2024 published CSVs."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 64:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(N // 4),
                                       12).astype(int))
    F = []
    for s in scales:
        n_seg = N // s
        if n_seg < 2:
            continue
        segs = y[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s, dtype=np.float64)
        A = np.column_stack([t, np.ones(s)])
        coef, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)
        resid = segs.T - A @ coef
        F.append(np.sqrt(np.mean(resid ** 2)))
    F = np.array(F)
    ok = F > 0
    if ok.sum() < 3:
        return np.nan
    slope = np.polyfit(np.log(scales[: len(F)][ok]), np.log(F[ok]), 1)[0]
    return float(slope)


# ----------------------------------------------------------------------
# Marginal complexity screen (folded-in 2024 fractal POC)
# ----------------------------------------------------------------------

def marginal_analysis(x, sr, window_sizes=(1.0, 3.0), step_s=0.1):
    """Sliding-window HFD/DFA over the raw waveform."""
    results = {"time": [], "hfd": {}, "dfa": {}}
    step = max(1, int(step_s * sr))
    for size in window_sizes:
        win = int(size * sr)
        hs, ds = [], []
        for i in range(0, len(x) - win + 1, step):
            w = x[i: i + win]
            hs.append(higuchi_fd(w))
            ds.append(dfa(w))
            if size == window_sizes[0]:
                results["time"].append(i / sr)
        results["hfd"][size] = hs
        results["dfa"][size] = ds
    return results


def adaptive_threshold(results):
    """Sample-adaptive thresholds: mean + 0.25 sigma over all windows.

    These thresholds are derived from the sample under test, so the
    marginal screen ranks windows WITHIN a recording; it cannot certify a
    whole recording on its own. The coherence screen's permutation test
    is the calibrated verdict. Retained from the original POC for
    continuity of method."""
    all_hfd = np.concatenate([results["hfd"][s] for s in results["hfd"]])
    all_dfa = np.concatenate([results["dfa"][s] for s in results["dfa"]])
    hfd_threshold = float(np.nanmean(all_hfd) + 0.25 * np.nanstd(all_hfd))
    dfa_threshold = float(np.nanmean(all_dfa) + 0.25 * np.nanstd(all_dfa))
    return hfd_threshold, dfa_threshold


def classify_windows(results, hfd_threshold, dfa_threshold):
    """Per-window synthetic/human vote (original POC logic: 5-window
    smoothing, HFD weighted 0.7 / DFA 0.3, high value OR high local
    variability votes synthetic)."""
    min_length = min(len(results["hfd"][s]) for s in results["hfd"])
    window, hfd_w, dfa_w = 5, 0.7, 0.3
    cls = []
    for i in range(max(0, min_length - window + 1)):
        hfd_avg = np.nanmean([np.nanmean(results["hfd"][s][i:i + window])
                              for s in results["hfd"]])
        dfa_avg = np.nanmean([np.nanmean(results["dfa"][s][i:i + window])
                              for s in results["dfa"]])
        hfd_var = np.nanmean([np.nanstd(results["hfd"][s][i:i + window])
                              for s in results["hfd"]])
        dfa_var = np.nanmean([np.nanstd(results["dfa"][s][i:i + window])
                              for s in results["dfa"]])
        hfd_ai = (hfd_avg > hfd_threshold) or (hfd_var > 0.1)
        dfa_ai = (dfa_avg > dfa_threshold) or (dfa_var > 0.05)
        cls.append(bool(hfd_w * hfd_ai + dfa_w * dfa_ai > 0.5))
    if cls:
        cls.extend([cls[-1]] * (window - 1))
    else:
        cls = [False] * min_length  # too few windows to smooth
    return cls


def write_marginal_csv(results, classifications, path):
    min_length = min(len(results["time"]), len(classifications),
                     *(len(results["hfd"][s]) for s in results["hfd"]))
    sizes = list(results["hfd"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Time", "Classification"]
                   + [f"HFD_{s}s" for s in sizes] + [f"DFA_{s}s" for s in sizes])
        for i in range(min_length):
            w.writerow([results["time"][i], classifications[i]]
                       + [results["hfd"][s][i] for s in sizes]
                       + [results["dfa"][s][i] for s in sizes])


def plot_marginal(results, hfd_threshold, dfa_threshold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    n = min(len(results["time"]),
            *(len(results["hfd"][s]) for s in results["hfd"]))
    for s in results["hfd"]:
        ax1.plot(results["time"][:n], results["hfd"][s][:n], label=f"HFD {s}s")
    ax1.axhline(y=hfd_threshold, color="r", linestyle="--", label="threshold")
    ax1.set_ylabel("Higuchi Fractal Dimension")
    ax1.legend()
    ax1.grid(True)
    for s in results["dfa"]:
        ax2.plot(results["time"][:n], results["dfa"][s][:n], label=f"DFA {s}s")
    ax2.axhline(y=dfa_threshold, color="r", linestyle="--", label="threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("DFA exponent")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def report_marginal(results, hfd_threshold, dfa_threshold, classifications):
    print("=== Marginal complexity screen (sliding-window HFD/DFA) ===")
    for s in results["hfd"]:
        h = np.array(results["hfd"][s], dtype=float)
        d = np.array(results["dfa"][s], dtype=float)
        print(f"  window {s}s: "
              f"HFD mean={np.nanmean(h):.4f} std={np.nanstd(h):.4f} "
              f"range=[{np.nanmin(h):.4f}, {np.nanmax(h):.4f}]   "
              f"DFA mean={np.nanmean(d):.4f} std={np.nanstd(d):.4f} "
              f"range=[{np.nanmin(d):.4f}, {np.nanmax(d):.4f}]")
    print(f"  adaptive thresholds: HFD {hfd_threshold:.4f}, "
          f"DFA {dfa_threshold:.4f} (sample-relative; see docstring)")
    n = len(classifications)
    if n:
        frac = float(np.mean(classifications))
        thirds = np.array_split(np.array(classifications, dtype=float), 3)
        thirds_s = ", ".join(f"{np.mean(t):.2f}" for t in thirds if len(t))
        label = "synthetic-leaning" if frac > 0.5 else "human-leaning"
        print(f"  windows voting synthetic: {frac:.2%} "
              f"(by thirds: {thirds_s})")
        print(f"  marginal screen result: {label} "
              f"(relative measure, not a standalone verdict)")
    print()


# ----------------------------------------------------------------------
# Narrative alignment
# ----------------------------------------------------------------------

def load_narrative(path):
    with open(path) as fh:
        data = json.load(fh)
    segs = data["segments"]
    for s in segs:
        assert "text" in s and "expected_arousal" in s, \
            "each segment needs text and expected_arousal"
    return segs


def segment_bounds(segs, audio_len, sr, timestamps=None):
    """Map narrative segments to sample ranges.

    With ASR word timestamps: exact. Without: proportional by word count
    across the trimmed audio (adequate for a read narrative at steady
    engagement; replace with forced alignment in production)."""
    if timestamps:
        bounds, w_idx = [], 0
        for s in segs:
            n_words = len(s["text"].split())
            start_w = timestamps[min(w_idx, len(timestamps) - 1)]
            end_w = timestamps[min(w_idx + n_words - 1, len(timestamps) - 1)]
            bounds.append((int(start_w["start"] * sr),
                           int(end_w["end"] * sr)))
            w_idx += n_words
        return bounds
    counts = np.array([len(s["text"].split()) for s in segs], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
    return [(int(edges[i] * audio_len), int(edges[i + 1] * audio_len))
            for i in range(len(segs))]


# ----------------------------------------------------------------------
# Coherence screen
# ----------------------------------------------------------------------

def analyze_segment(x, sr):
    frames, _ = frame_signal(x, sr)
    if len(frames) < 8:
        return None
    f0, energies = f0_autocorr(frames, sr)
    voiced = ~np.isnan(f0)
    env = energies - energies.mean()
    return {
        "f0_mean": float(np.nanmean(f0)) if voiced.any() else np.nan,
        "jitter": jitter(f0),
        "shimmer": shimmer(energies, voiced),
        "rate": onset_rate(x, sr),
        "pause_ratio": float(1.0 - voiced.mean()),
        "hfd": higuchi_fd(env),
        "dfa": dfa(env),
    }


AROUSAL_KEYS = ["jitter", "shimmer", "f0_mean", "rate"]

# Per-feature modulation floors: the minimum absolute range a proxy must
# span across the narrative before its variation counts as physiological
# modulation rather than measurement noise. Motivated by typical human
# phonation values (jitter and shimmer are ratios; F0 in Hz; rate in
# Hz of syllabic modulation). A voice whose proxies stay under these
# floors across a calm-to-peak narrative shows no stress response at all,
# which is itself the synthesis signature.
RANGE_FLOORS = {"jitter": 0.003, "shimmer": 0.015,
                "f0_mean": 6.0, "rate": 0.7}
MIN_ACTIVE_FEATURES = 2


def coherence(segs, feats, n_perm=5000, rng=None):
    expected = np.array([s["expected_arousal"] for s in segs], dtype=float)
    mat, active = [], []
    for k in AROUSAL_KEYS:
        col = np.array([f[k] if f and not np.isnan(f[k]) else np.nan
                        for f in feats], dtype=float)
        if np.isnan(col).all():
            continue
        col = np.where(np.isnan(col), np.nanmean(col), col)
        rng_span = float(np.max(col) - np.min(col))
        if rng_span < RANGE_FLOORS[k] or np.std(col) == 0:
            continue  # flat proxy: no evidence either way
        z = np.clip(zscore(col), -2.5, 2.5)
        mat.append(z)
        active.append(k)
    depth = len(active)
    if depth < MIN_ACTIVE_FEATURES:
        return np.nan, np.nan, None, depth, active, {}
    # per-feature correlations: the composite can mask a proxy moving
    # against the others (jitter/shimmer direction under stress is
    # speaker-dependent in the phonation literature)
    per_feature = {k: float(spearmanr(expected, z)[0])
                   for k, z in zip(active, mat)}
    observed = np.mean(np.stack(mat), axis=0)
    rho, _ = spearmanr(expected, observed)
    rng = rng or np.random.default_rng(0)
    null = np.array([spearmanr(rng.permutation(expected), observed)[0]
                     for _ in range(n_perm)])
    # add-one correction: a permutation p-value cannot honestly be zero
    p = float((np.sum(null >= rho) + 1) / (n_perm + 1))
    return float(rho), p, observed, depth, active, per_feature


def run(audio_path, narrative_path=None, seconds=60.0, ts_path=None,
        csv_path=None, plot_path=None, step_s=0.1):
    sr, x = load_audio(audio_path, seconds)

    mres = marginal_analysis(x, sr, step_s=step_s)
    hfd_t, dfa_t = adaptive_threshold(mres)
    cls = classify_windows(mres, hfd_t, dfa_t)
    report_marginal(mres, hfd_t, dfa_t, cls)
    if csv_path:
        write_marginal_csv(mres, cls, csv_path)
        print(f"Marginal results written to {csv_path}")
    if plot_path:
        plot_marginal(mres, hfd_t, dfa_t, plot_path)
        print(f"Marginal plot written to {plot_path}")

    if not narrative_path:
        return None, None
    segs = load_narrative(narrative_path)
    timestamps = None
    if ts_path:
        with open(ts_path) as fh:
            timestamps = json.load(fh)
    bounds = segment_bounds(segs, len(x), sr, timestamps)
    feats = [analyze_segment(x[a:b], sr) for a, b in bounds]
    rho, p, observed, depth, active, per_feature = coherence(segs, feats)
    report(segs, feats, rho, p, observed, depth, active, per_feature)
    return rho, p


def report(segs, feats, rho, p, observed, depth, active, per_feature):
    print("=== Narrative stress-coherence screen ===")
    hdr = (f"{'seg':<4}{'exp_arousal':>12}{'obs_arousal':>12}{'F0':>8}"
           f"{'jitter':>9}{'shimmer':>9}{'rate':>7}{'HFD':>7}{'DFA':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, (s, f) in enumerate(zip(segs, feats)):
        if f is None:
            print(f"{i:<4}{'(segment too short)':>12}")
            continue
        obs = observed[i] if observed is not None else float("nan")
        print(f"{i:<4}{s['expected_arousal']:>12.2f}{obs:>12.2f}"
              f"{f['f0_mean']:>8.1f}{f['jitter']:>9.4f}"
              f"{f['shimmer']:>9.4f}{f['rate']:>7.2f}"
              f"{f['hfd']:>7.3f}{f['dfa']:>7.3f}")
    print("-" * len(hdr))
    print(f"Stress-coherence (Spearman rho): {rho:+.3f}   "
          f"permutation p = {p:.4f}   "
          f"active proxies = {depth} {active}")
    if per_feature:
        print("Per-feature rho: " + "  ".join(
            f"{k} {v:+.3f}" for k, v in per_feature.items()))
    # Illustrative decision logic; calibrate on real corpora.
    if depth < MIN_ACTIVE_FEATURES:
        verdict = ("INCOHERENT: stress proxies are flat across a "
                   "calm-to-peak narrative (no physiological modulation; "
                   "synthesis signature regardless of any correlation)")
    elif np.isnan(rho):
        verdict = "INDETERMINATE (insufficient voiced signal)"
    elif rho >= 0.6 and p <= 0.05:
        verdict = ("COHERENT: perturbations track the narrative's stress "
                   "profile (consistent with a human processing content)")
    elif rho <= 0.15:
        verdict = ("INCOHERENT: perturbations are decoupled from content "
                   "(consistent with synthesis or flat read)")
    else:
        verdict = "WEAK COHERENCE: collect more segments or a longer sample"
    print(f"Verdict: {verdict}")
    print("Note: the coherence screen complements the marginal HFD/DFA "
          "screen; run both, and only after positive-detection identity "
          "checks have passed.")


# ----------------------------------------------------------------------
# Self-test: synthesize a 'human' whose perturbations track the arousal
# curve and a 'synthetic' flat reader; the analyzer must separate them.
# Also smoke-tests the folded-in marginal screen.
# ----------------------------------------------------------------------

def synth_voice(sr, segs, coupled, seed, decorated=False):
    rng = np.random.default_rng(seed)
    out = []
    for s in segs:
        dur = 0.28 * len(s["text"].split()) + 0.4
        n = int(dur * sr)
        t = np.arange(n) / sr
        if coupled:
            a = s["expected_arousal"]
        elif decorated:
            a = rng.uniform(0.1, 0.95)  # rich variation, content-blind
        else:
            a = 0.5
        modulated = coupled or decorated
        f0 = 110 + 40 * a  # stress raises pitch
        jit = (0.004 + 0.025 * a) if modulated else 0.004
        # cycle-scale jitter: low-pass modulation noise, as in phonation,
        # rather than per-sample white FM which averages out
        slow = rng.standard_normal(n)
        kj = np.hanning(max(3, int(sr / 60)))
        slow = np.convolve(slow, kj / kj.sum(), mode="same")
        slow = slow / (np.std(slow) or 1.0)
        phase = 2 * np.pi * np.cumsum(f0 * (1 + jit * slow)) / sr
        carrier = np.sign(np.sin(phase)) * 0.6 + 0.4 * np.sin(2 * phase)
        # syllabic amplitude modulation; rate rises with arousal if coupled
        syl = 3.0 + (2.5 * a if modulated else 1.25)
        am = 0.55 + 0.45 * np.clip(np.sin(2 * np.pi * syl * t), 0, None)
        shim = (0.05 + 0.30 * a) if modulated else 0.08
        am *= 1 + shim * rng.standard_normal(n) * 0.1
        seg_audio = carrier * am
        if not modulated:  # smooth generator-style regularity
            k = np.hanning(int(0.004 * sr))
            seg_audio = np.convolve(seg_audio, k / k.sum(), mode="same")
        out.append(seg_audio)
        out.append(np.zeros(int(0.15 * sr)))
    audio = np.concatenate(out)
    # ground-truth bounds (equivalent to supplying ASR word timestamps;
    # the proportional mapping is exercised separately in field use)
    bounds, pos = [], 0
    for i in range(0, len(out), 2):
        bounds.append((pos, pos + len(out[i])))
        pos += len(out[i]) + len(out[i + 1])
    return audio, bounds


def selftest():
    sr = 16000
    segs = [
        {"text": "the morning was quiet and the coffee was warm",
         "expected_arousal": 0.10},
        {"text": "a routine drive along a road you have taken many times",
         "expected_arousal": 0.20},
        {"text": "the phone rings once and then stops ringing",
         "expected_arousal": 0.35},
        {"text": "a name you have not heard in years appears on the screen",
         "expected_arousal": 0.55},
        {"text": "the door at the end of the hall is standing open",
         "expected_arousal": 0.75},
        {"text": "someone is inside the house and they know your name",
         "expected_arousal": 0.95},
        {"text": "you breathe and count backwards slowly from ten",
         "expected_arousal": 0.45},
        {"text": "the room is empty and the window was only the wind",
         "expected_arousal": 0.15},
    ]
    print("=== SELF-TEST: coupled (human-like) speaker ===")
    xh, bh = synth_voice(sr, segs, coupled=True, seed=7)
    fh = [analyze_segment(xh[a:b], sr) for a, b in bh]
    rho_h, p_h, obs_h, d_h, a_h, pf_h = coherence(segs, fh)
    report(segs, fh, rho_h, p_h, obs_h, d_h, a_h, pf_h)
    print()
    print("=== SELF-TEST: decoupled (generator-like) reader ===")
    xs, bs = synth_voice(sr, segs, coupled=False, seed=7)
    fs = [analyze_segment(xs[a:b], sr) for a, b in bs]
    rho_s, p_s, obs_s, d_s, a_s, pf_s = coherence(segs, fs)
    report(segs, fs, rho_s, p_s, obs_s, d_s, a_s, pf_s)
    print()
    coupled_pass = (d_h >= MIN_ACTIVE_FEATURES) and (rho_h >= 0.6)
    decoupled_fail = (d_s < MIN_ACTIVE_FEATURES) or \
        (not np.isnan(rho_s) and rho_s < 0.15)
    print()
    print("=== SELF-TEST: decorated generator (variation, content-blind) ===")
    xd, bd = synth_voice(sr, segs, coupled=False, seed=11, decorated=True)
    fd = [analyze_segment(xd[a:b], sr) for a, b in bd]
    rho_d, p_d, obs_d, d_d, a_d, pf_d = coherence(segs, fd)
    report(segs, fd, rho_d, p_d, obs_d, d_d, a_d, pf_d)
    print()
    decorated_fail = (d_d < MIN_ACTIVE_FEATURES) or \
        (not np.isnan(rho_d) and rho_d < 0.6)
    print("=== SELF-TEST: marginal screen smoke (folded fractal POC) ===")
    mres = marginal_analysis(xh, sr, step_s=0.5)
    hfd_t, dfa_t = adaptive_threshold(mres)
    cls = classify_windows(mres, hfd_t, dfa_t)
    report_marginal(mres, hfd_t, dfa_t, cls)
    marginal_ok = (len(cls) > 0 and np.isfinite(hfd_t) and np.isfinite(dfa_t))
    ok = coupled_pass and decoupled_fail and decorated_fail and marginal_ok
    print(f"SELF-TEST {'PASSED' if ok else 'FAILED'}: "
          f"coupled rho={rho_h:+.3f}/proxies={d_h}, "
          f"flat proxies={d_s}, "
          f"decorated rho={rho_d if not np.isnan(rho_d) else float('nan'):+.3f}"
          f"/proxies={d_d}, "
          f"marginal windows={len(cls)}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="VoiceKey voice authenticity analyzer "
                    "(marginal HFD/DFA screen + narrative stress coherence)")
    ap.add_argument("audio", nargs="?", help=".wav file")
    ap.add_argument("narrative", nargs="?",
                    help="narrative annotation JSON (enables the "
                         "coherence screen)")
    ap.add_argument("--timestamps", help="optional ASR word timestamps JSON")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=0.1,
                    help="marginal screen hop in seconds")
    ap.add_argument("--csv", help="write per-window marginal results CSV")
    ap.add_argument("--plot", help="write marginal plot PNG "
                                   "(requires matplotlib)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.audio:
        ap.error("an audio file is required (or use --selftest)")
    run(args.audio, args.narrative, args.seconds, args.timestamps,
        args.csv, args.plot, args.step)


if __name__ == "__main__":
    main()
