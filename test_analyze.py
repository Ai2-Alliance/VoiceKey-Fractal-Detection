"""Smoke test for the merged analyzer (coherence_analyze.py).

Runs the built-in selftest (synthetic coupled / flat / decorated voices
plus the marginal-screen smoke), then the marginal screen against the
repo's human test sample. Exit code 0 means both completed."""

import subprocess
import sys

r1 = subprocess.run([sys.executable, "coherence_analyze.py", "--selftest"])
r2 = subprocess.run([sys.executable, "coherence_analyze.py",
                     "test-samples/voicekey-test1-human.wav",
                     "--seconds", "10", "--step", "0.5"])
code = r1.returncode or r2.returncode
print("Test completed." if code == 0 else "Test FAILED.")
sys.exit(code)
