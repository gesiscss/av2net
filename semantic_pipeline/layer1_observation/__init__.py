"""Layer 1 - Observation.

Transforms an MP4 into temporally aligned multimodal Observation objects via
speech recognition, OCR, and visual interpretation. No semantic network is
constructed here; observations are ephemeral inputs to Layer 2.

Each capability sits behind an abstract interface (see `interfaces`). Local and
production implementations satisfy the same interface, so the pipeline logic is
model-agnostic.
"""
