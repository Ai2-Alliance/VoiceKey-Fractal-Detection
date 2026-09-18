# Fractal Analysis for VoiceKey

## An Open-Source Research Project in Support of AI Integrity Alliance

## Project Overview

This open-source research project develops voice authenticity analysis for the VoiceKey initiative, conducted in support of the AI Integrity Alliance's (AI²) mission to create robust methods for voice authenticity verification in an era of advancing AI voice synthesis.

The project is one evolving analyzer, `coherence_analyze.py`, providing two complementary **negative-detection** screens over a single recording:

1. **Marginal complexity screen** — sliding-window Higuchi Fractal Dimension (HFD) and Detrended Fluctuation Analysis (DFA) over the raw waveform, with sample-adaptive thresholds and per-window classification. This is the original fractal proof-of-concept, folded in.
2. **Narrative stress-coherence screen** — given a one-shot challenge narrative annotated with an expected-arousal profile, measures whether the speaker's physiological perturbations (jitter, shimmer, pitch, tempo) *track the semantic stress points of the content* — the conditional signal a psyche processing content through a body emits for free.

The marginal screen measures the distribution of signal complexity; the coherence screen measures perturbation *coupled to content*. They reject different failure modes; run both.

### Position in the VoiceKey Pipeline

This module is **negative detection** (see the VoiceKey [negative detection doc](https://github.com/Ai2-Alliance/VoiceKey/blob/main/docs/negative_detection.md)) and runs **only after all positive-detection identity mechanics have passed**: biometric voice match, MFA, liveness, session binding. Positive detection answers "is this the enrolled identity?"; this layer answers "is any signature of synthesis present?" — the *absence* of human analog signatures is the rejection signal. It contributes nothing to identification and must never be used as a standalone authenticator.

## Understanding HFD and DFA

- **HFD**: The Higuchi Fractal Dimension measures how complex and detailed a signal is at different scales.
- **DFA**: Detrended Fluctuation Analysis measures how a signal's fluctuations scale with observation window size after removing local trends, revealing long-range correlations.

Both are applied two ways: sliding windows over the raw waveform (marginal screen) and per narrative segment over the energy envelope (coherence screen supporting statistics).

## The Rationale: Computational Exhaustion vs Pure Analog

The security claim is **not** that these signatures cannot be synthesized. A resourced generator driving an emotion-controllable vocoder from an LLM arousal annotation of the challenge text can pass a shallow version of the coherence check; that bypass is acknowledged and priced in. The claim is **asymmetric cost**, in the spirit of proof-of-work:

- **A human produces both signals for free.** They are analog side effects of a body phonating and a psyche processing content in real time — zero compute, zero preparation, at any depth and any duration.
- **A generator must simulate them explicitly**: match the marginal complexity statistics, estimate the arousal profile, render perturbations onto it, and keep every proxy (pitch, perturbation, tempo, pausing, and their co-movement) mutually consistent for the full sample, one-shot, in real time.

Attack cost therefore scales with challenge **depth** — segment count, sample duration, number of proxies, annotation subtlety, cross-proxy consistency — while verification cost and the legitimate speaker's cost stay flat. Depth is the security knob, tuned to the value of the unlock it gates, exactly as work factors are tuned in password hashing: a low-value unlock warrants a short, shallow narrative; a high-value unlock warrants long, deep, multi-axis coherence, at which point the generator is doing substantial real-time modeling work to imitate what the analog system emits as exhaust.

What defeats the shallow attacks outright: replayed recordings of other content and content-blind synthesis (flat reads, or decorative variation uncoupled to the narrative) both fail the coherence screen — see the [comparison analysis](fractal_analysis_comparison.md) for measured results.

## Findings To Date

Summary of the human vs ElevenLabs comparison (full details, tables, and adversarial testing in [fractal_analysis_comparison.md](fractal_analysis_comparison.md)):

- The AI-generated sample shows **higher and more variable HFD and DFA** than the human sample at both 1 s and 3 s windows — a distributional difference consistent across the 2024 and 2026 analysis runs.
- On the coherence screen, reading the same annotated narrative: the human sample scores **WEAK coherence** (rho = +0.381) and the ElevenLabs sample scores **INCOHERENT** (rho = +0.048) — notable because both recordings were made under "maintain a consistent tone" instructions, which *suppress* the coherence signal in the human read. Recordings made under natural-affect instructions are the next validation step.
- The analyzer's built-in selftest separates a coupled (human-like) synthetic speaker (rho = +1.000), a flat reader (0 active proxies), and a content-blind decorated generator (rho = +0.357) — proof the mechanism is measurable.

These findings remain preliminary and based on a minimal corpus.

## Getting Started

### Prerequisites

- Python 3.9+
- numpy, scipy (matplotlib only if you want `--plot` output)

### Installation

```
git clone https://github.com/Ai2-Alliance/VoiceKey-Fractal-Detection.git
cd VoiceKey-Fractal-Detection
pip install numpy scipy
```

### Usage

Marginal screen only:

```
python coherence_analyze.py path/to/audio.wav
```

Both screens (narrative annotation enables coherence):

```
python coherence_analyze.py path/to/audio.wav narrative.json
```

Options:

```
--seconds 60        analysis window length (default 60)
--step 0.1          marginal screen hop in seconds
--timestamps w.json optional ASR word timestamps for exact segment alignment
--csv out.csv       write per-window marginal results
--plot out.png      write marginal plot (requires matplotlib)
--selftest          run the built-in synthetic separation test
```

A full 60-second sample analyzes in roughly 10 seconds on commodity hardware.

### Narrative Annotation Format

The coherence screen needs the challenge narrative split into ordered segments, each with the text read aloud and an expected arousal value in [0, 1]:

```json
{
  "segments": [
    {"text": "the morning was quiet and the coffee was warm", "expected_arousal": 0.10},
    {"text": "someone is inside the house and they know your name", "expected_arousal": 0.95}
  ]
}
```

Segments are mapped onto the audio proportionally by word count, or exactly if ASR word timestamps are supplied. A complete example — the annotation of the project's own test narrative — is at [voicekey-test-narrative.json](voicekey-test-narrative.json):

```
python coherence_analyze.py test-samples/voicekey-test1-human.wav voicekey-test-narrative.json
```

## Test Narrative

The original test narrative is at [voicekey-test-narrative.md](voicekey-test-narrative.md). Note its instruction to "maintain a consistent tone" is appropriate for the **marginal** screen only — coherence challenges require the opposite instruction (read naturally, let the content land), because a deliberately flat read suppresses exactly the signal the coherence screen measures.

## Future Work and Crowdsourcing Initiative

1. **Sample collection**: diverse human and AI-generated recordings of arousal-annotated narratives, recorded under natural-affect instructions.
2. **Threshold calibration**: the coherence verdict bands (rho/p cutoffs, per-feature modulation floors) are illustrative and need calibration on a real corpus.
3. **Alignment**: replace proportional word-count segment mapping with forced alignment / ASR word timestamps for field use.
4. **Adversarial testing**: measure the actual cost curve for emotion-conditioned synthesis to pass at increasing challenge depth.

If you're interested in contributing, please check our [Contributing Guidelines](https://ai2-alliance.github.io/VoiceKey/docs/collaboration.html).

## VoiceKey Project

VoiceKey is a research initiative aimed at developing a robust voice authentication system leveraging the unique randomness properties of the human voice. It utilizes negative detection, Zero-Knowledge Proofs (ZKPs), blockchain technology, and analog voice verification to create a secure, privacy-preserving, and computationally efficient authentication mechanism. [VoiceKey Project Overview](https://ai2-alliance.github.io/VoiceKey/)

Key aspects of VoiceKey include:
- Negative detection of AI-generated voices
- Integration of Multi-Factor Authentication (MFA) and biometric factors
- Privacy preservation using ZKP and blockchain technology
- Analysis of compute resource expenditures and bypass probabilities
- Consideration of potential bypass methodologies and security measures

## AI Integrity Alliance (AI²)

The AI Integrity Alliance (AI²) is a global coalition dedicated to promoting ethical and trustworthy artificial intelligence. Its mission is to ensure AI technologies are developed and used responsibly, transparently, and inclusively.

Core Principles of AI²:
1. Transparency and Accountability
2. Inclusion and Diversity
3. Open-Source Empowerment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is a research initiative and should not be considered a definitive method for distinguishing human from AI-generated voices. The effectiveness of this approach may vary and is subject to ongoing research and refinement.

## Acknowledgments

- This project is conducted in support of the [AI Integrity Alliance](https://ai2.ngo)
- Special thanks to all contributors and researchers in the field of audio signal processing and fractal analysis

## Contact

For inquiries about this project or the AI Integrity Alliance, please contact:
[info@ai2.ngo](mailto:info@ai2.ngo)

---

## Version Log

Single evolving analyzer — no version branching; this log records the evolution.

- **2026-09-18 — Coherence evolution.**
  - Added the narrative stress-coherence screen: per-segment jitter / shimmer / F0 / articulation-rate proxies correlated (Spearman + permutation p, per-feature breakdown) against a challenge narrative's expected-arousal profile; verdict bands COHERENT / WEAK / INCOHERENT / flat-proxy synthesis signature.
  - Folded the original fractal POC (`analyze.py`) into `coherence_analyze.py` as the marginal complexity screen; `analyze.py` retired.
  - Corrected the Higuchi normalization to the standard definition (legacy slopes were shifted by ~1; 2024 CSV values are internally consistent but not comparable to current output).
  - Vectorized DFA detrending: a full 60 s sample now analyzes in ~10 s instead of hours.
  - Robustness fixes: voiced-run-restricted jitter/shimmer (pause boundaries no longer inject spurious perturbation), silence-trim off-by-one, short-segment and silent-input edge cases, add-one permutation p correction.
  - Dependencies reduced to numpy + scipy (matplotlib optional, `--plot` only); CSV/PNG output now opt-in flags.
  - Reframed the security thesis: computational exhaustion vs pure analog, with challenge depth as the work factor, positioned explicitly as negative detection after positive-detection identity mechanics.
- **2024-10-14 — Initial fractal POC.**
  - Sliding-window HFD/DFA analyzer (`analyze.py`) with sample-adaptive thresholds, CSV/plot output.
  - Human vs ElevenLabs test corpus recorded from the shared test narrative; initial comparison findings published ([fractal_analysis_comparison.md](fractal_analysis_comparison.md)).
