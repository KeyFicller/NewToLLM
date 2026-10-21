import pytest

from conftest import REPO_ROOT

EXE = REPO_ROOT / "build" / "cpp_impl" / "tools" / "parity_check"


@pytest.mark.skipif(
    not EXE.exists(), reason="C++ side not built (build/cpp_impl/tools/parity_check)"
)
@pytest.mark.parametrize("seed", [123, 7])
def test_small_model_parity(seed):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from parity_check import run_parity

    ok, max_abs = run_parity(full=False, seed=seed, exe=EXE)
    assert ok, f"max_abs_diff={max_abs}"
