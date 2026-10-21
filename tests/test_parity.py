"""Gate: same seed => same weights, same weights => same logits (Python vs C++)."""

from __future__ import annotations

import pytest

from parity import PARITY_EXE, run


@pytest.mark.skipif(not PARITY_EXE.exists(), reason="C++ side not built")
@pytest.mark.parametrize("seed", [123, 7])
def test_tiny_and_small_presets_agree(seed: int) -> None:
    results = run(["tiny", "small"], seed)
    failed = [r for r in results if not r.passed]
    assert not failed, failed
    assert {r.check for r in results} == {"init", "forward"}
