# Voice Authenticity Analysis: Human vs AI-Generated Audio

## Introduction

This document presents the comparison of human and AI-generated audio samples under the project's two negative-detection screens: the marginal complexity screen (sliding-window Higuchi Fractal Dimension and Detrended Fluctuation Analysis) and the narrative stress-coherence screen (physiological perturbation proxies correlated against the challenge narrative's expected-arousal profile).

Both samples are 60-second reads of the same test narrative ([voicekey-test-narrative.md](voicekey-test-narrative.md)):

1. Human recording (`test-samples/voicekey-test1-human.wav`)
2. ElevenLabs AI generation (`test-samples/voicekey-test1-ai-elevenlabs.wav`)

This remains an exploratory analysis on a minimal corpus; conclusions are directional, not definitive.

## Marginal Complexity Screen — 2026 Run (corrected normalization)

Produced by `coherence_analyze.py` with default settings (1 s and 3 s windows, 0.1 s hop). These values use the **standard Higuchi normalization** and are NOT comparable to the 2024 numbers below (see Methodology Notes).

### Higuchi Fractal Dimension (HFD)

| Window | Sample | Mean | Std | Range |
|--------|--------|------|-----|-------|
| 1 s | Human | 1.2900 | 0.0821 | [1.0869, 1.5529] |
| 1 s | AI | 1.4420 | 0.1345 | [1.1056, 1.9023] |
| 3 s | Human | 1.2837 | 0.0362 | [1.1663, 1.3736] |
| 3 s | AI | 1.4546 | 0.0622 | [1.2838, 1.6477] |

### Detrended Fluctuation Analysis (DFA)

| Window | Sample | Mean | Std | Range |
|--------|--------|------|-----|-------|
| 1 s | Human | 0.7927 | 0.0539 | [0.6646, 1.1570] |
| 1 s | AI | 0.9226 | 0.0713 | [0.7848, 1.1621] |
| 3 s | Human | 0.7478 | 0.0298 | [0.6643, 1.0844] |
| 3 s | AI | 0.8171 | 0.0696 | [0.7136, 0.9744] |

### Observations

1. **Direction confirmed under corrected normalization**: the AI sample shows higher and more variable HFD and DFA at both window sizes, matching the direction of the 2024 findings. At the 3 s window the HFD means are separated by more than two of the human sample's standard deviations.
2. **Cross-sample statistics separate; self-relative classification does not.** The per-window vote fractions are similar for both files (human 39.75% synthetic-voting windows, AI 43.83%) because the adaptive thresholds are derived from the sample under test — they rank windows *within* a recording and cannot certify a recording on their own. This is a structural limitation of self-referenced thresholds, now documented in the analyzer. Certifying a recording from marginal statistics requires externally calibrated thresholds from a real corpus.
3. **Runtime**: ~9 s per 60-second 44.1 kHz sample (vectorized DFA), versus hours in the 2024 POC. Compute-cost-as-security now rests on challenge depth (see README rationale), not on analysis slowness.

## Narrative Stress-Coherence Screen — 2026 Run

The test narrative was annotated with an expected-arousal profile (8 segments; values below) and both recordings were scored against it. **Important caveat**: both recordings were made under the original instruction to "maintain a consistent tone," which suppresses the coherence signal in the human read. This run is therefore an illustration of the mechanism, not a validation of human-pass rates.

Segment arousal annotation used: [voicekey-test-narrative.json](voicekey-test-narrative.json) — 0.25, 0.35, 0.55, 0.45, 0.40, 0.65, 0.60, 0.75 (one value per sentence/clause group of the narrative, in order; proportional word-count alignment).

| Sample | Spearman rho | Permutation p | Active proxies | Verdict |
|--------|--------------|---------------|----------------|---------|
| Human | +0.381 | 0.178 | 4 (jitter, shimmer, F0, rate) | WEAK COHERENCE |
| AI (ElevenLabs) | +0.048 | 0.466 | 4 (jitter, shimmer, F0, rate) | INCOHERENT |

Per-feature correlations:

| Sample | jitter | shimmer | F0 mean | rate |
|--------|--------|---------|---------|------|
| Human | −0.190 | +0.667 | +0.286 | −0.095 |
| AI | +0.310 | +0.405 | +0.429 | −0.310 |

### Observations

1. Even under tone-suppressing instructions, the human read retains partial content coupling (shimmer +0.667) while the AI generation is fully decoupled (composite rho +0.048, near the permutation null).
2. Neither sample reaches the COHERENT band (rho ≥ 0.6, p ≤ 0.05) — expected, given the flat-tone instruction. The decisive experiment is a recording session under natural-affect instructions with a calm-to-peak narrative.
3. The per-feature signs differ between proxies and between samples, consistent with the phonation literature: stress-direction of jitter/shimmer is speaker-dependent. This is why the analyzer reports per-feature correlations rather than trusting the composite alone.

## Built-in Selftest: Three-Way Separation

The analyzer synthesizes three speakers over a calm-to-peak narrative and must separate them (`python coherence_analyze.py --selftest`):

| Case | Result | Verdict |
|------|--------|---------|
| Coupled (perturbations track arousal) | rho = +1.000, p = 0.0002, 4 proxies | COHERENT |
| Flat reader (no modulation) | 0 active proxies | INCOHERENT (flat-proxy synthesis signature) |
| Decorated (rich variation, content-blind) | rho = +0.357, p = 0.197, 4 proxies | WEAK |

This proves the measurement chain recovers coupling when present, rejects its absence, and is not fooled by variation alone.

## Adversarial Testing

Findings from adversarial probing of the coherence screen (2026):

1. **Content-blind variation rarely passes by luck.** Across 200 random seeds, the decorated (content-blind) generator crossed the COHERENT gate 9/200 times (~4.5%) — consistent with the p ≤ 0.05 gate. More segments per challenge lowers this floor.
2. **An informed adversary passes a shallow challenge.** A generator that estimates the narrative's arousal profile (an LLM call or basic sentiment analysis) and drives synthesis controls with it passed the 8-segment COHERENT gate 50/50 with a clean estimate, and 34/50 even with heavy noise (σ = 0.2) on the estimate. **This is expected and priced in**: the expected-arousal profile is a function of the public challenge text by construction. The defense is not secrecy — it is that the attacker's real-time modeling and rendering cost scales with challenge depth (segments, duration, proxies, cross-proxy consistency) while the human's cost stays at zero. See the README rationale.
3. **Replay and flat reads fail structurally.** A replayed recording of different content, a flat TTS read, or decorative-but-uncoupled variation all land INCOHERENT or WEAK regardless of audio quality.

## Methodology Notes and 2024 Comparability

- **Higuchi normalization correction (2026)**: the 2024 POC (`analyze.py`, since retired) omitted the final /k in the Higuchi curve-length normalization, shifting its slope by ~1 relative to the standard definition. The 2024 CSVs (`fractal_analysis_results_*.csv`) and plots are internally consistent — the human/AI *comparison* they support stands — but their absolute values are not comparable to current output.
- **DFA scale selection** also changed (12 log-spaced scales, 8 to N/4, vs the POC's dense 5–100 range), and DFA detrending is now vectorized (numerically identical to the loop implementation, verified to machine precision).
- The 2024 published values for reference, on the legacy scale: HFD 3 s — human mean 0.2852, AI mean 0.4544; DFA 3 s — human mean 1.5159, AI mean 1.5898 (full tables in git history). Direction matches the 2026 run.
- Jitter and shimmer are frame-level perturbation proxies (10 ms hop, restricted to contiguous voiced runs), not Praat cycle-to-cycle measures; treat them as relative modulation indicators.

## Limitations

1. **Corpus of two.** Every number above is a single-sample observation per class.
2. **Coherence thresholds are illustrative.** The verdict bands (rho ≥ 0.6, p ≤ 0.05; rho ≤ 0.15) need calibration on a real corpus of natural-affect human reads before any operational meaning attaches to them.
3. **8 segments is near the statistical floor.** With n = 8, the COHERENT band sits close to the Spearman critical value; production challenges need more segments or repeated trials.
4. **Alignment is proportional.** Word-count mapping is adequate for steady reads; field use needs forced alignment or ASR word timestamps (supported via `--timestamps`).
5. **Reader instructions matter.** Flat-tone instructions suppress the human coherence signal; the existing recordings under-represent achievable human coherence.

## Future Directions

1. Record a natural-affect corpus: multiple humans reading calm-to-peak annotated narratives, plus emotion-conditioned AI generations of the same narratives — the head-to-head that measures whether the coherence gap survives a knowing adversary, and at what challenge depth.
2. Calibrate verdict bands and per-feature modulation floors on that corpus; replace sample-adaptive marginal thresholds with corpus-calibrated ones.
3. Add forced alignment for exact segment boundaries.
4. Measure the attacker cost curve: emotion-conditioned synthesis quality vs challenge depth (segments, duration, proxy count), to put numbers on the computational-exhaustion thesis.
5. Explore cross-proxy consistency checks (do jitter, shimmer, F0, tempo, and pausing co-move the way human phonation co-moves under load?) as additional depth axes that are cheap to verify and expensive to co-simulate.
