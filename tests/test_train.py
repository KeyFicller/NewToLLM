"""Gate: both sides build the same batches and follow the same loss curve."""

from __future__ import annotations

import pytest

from train_parity import CPP_CLI, compare


@pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ side not built")
def test_tiny_training_curves_match() -> None:
    result, py, _ = compare("tiny", 123)
    assert result.python_steps == result.cpp_steps == len(py["steps"]) > 0
    assert result.passed, result
