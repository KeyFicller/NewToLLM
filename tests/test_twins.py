"""The @twin/@step lint accepts the repo and rejects mismatched step anchors."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_twins

PY = "# @twin f\ndef f():\n    # @step a: 第一步\n    x = 1\n    # @step b: 第二步\n    return x\n"
CPP = "// @twin f\nint f() {\n  // @step A\n  int x = 1;\n  // @step B\n  return x;\n}\n"


def cpp(first, second):
    return CPP.replace("A", first).replace("B", second)


def test_repo_passes():
    assert check_twins.check() == []


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / "py").mkdir()
    (tmp_path / "cpp").mkdir()
    monkeypatch.setattr(check_twins, "PY_DIR", tmp_path / "py")
    monkeypatch.setattr(check_twins, "CPP_DIR", tmp_path / "cpp")
    monkeypatch.setattr(check_twins, "LESSONS", ("m",))

    def write(py, cpp):
        (tmp_path / "py" / "m.py").write_text(py)
        (tmp_path / "cpp" / "m.h").write_text("")
        (tmp_path / "cpp" / "m.cpp").write_text(cpp)
        return check_twins.check()

    return write


def test_matching_steps(tree):
    assert tree(PY, cpp("a", "b")) == []


def test_reordered_steps_fail(tree):
    assert any("step ids differ" in e for e in tree(PY, cpp("b", "a")))


def test_missing_note_fails(tree):
    assert any("needs a note" in e for e in tree(PY.replace(": 第一步", ""), cpp("a", "b")))


def test_lesson_without_steps_fails(tree):
    py = "# @twin f\ndef f():\n    return 1\n"
    assert any("no '# @step'" in e for e in tree(py, "// @twin f\nint f() { return 1; }\n"))
