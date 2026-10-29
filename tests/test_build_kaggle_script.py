"""Packaging tests: verify the bundled Kaggle script without any GPU.

These tests prove requirement 3 of the repository-hardening pass: the
build is deterministic, the output is valid Python, it contains the real
logic from every bundled module (not stubs), and running it with
--self-test succeeds as a subprocess using only this machine's CPU-only
interpreter -- no torch/transformers/datasets needed for the self-test
path itself (it degrades gracefully when those are absent).
"""

import subprocess
import sys

import build_kaggle_script


def test_build_is_deterministic_for_same_sources():
    first = build_kaggle_script.build()
    second = build_kaggle_script.build()
    assert first == second


def test_build_output_is_valid_python():
    source = build_kaggle_script.build()
    compile(source, "<generated-kaggle-script>", "exec")


def test_build_contains_every_module_marker():
    source = build_kaggle_script.build()
    for module_name in build_kaggle_script.MODULE_ORDER:
        assert f"src/clinasr/{module_name}.py" in source


def test_build_contains_real_logic_not_stubs():
    source = build_kaggle_script.build()
    assert "def normalize_text" in source
    assert "def add_noise_at_snr" in source
    assert "def match_entities" in source
    assert "class ExperimentConfig" in source


def test_build_has_single_future_import_at_top():
    source = build_kaggle_script.build()
    lines = source.splitlines()
    future_lines = [i for i, l in enumerate(lines) if l.strip() == "from __future__ import annotations"]
    assert len(future_lines) == 1
    # Nothing but the module docstring and a blank line should precede it.
    assert future_lines[0] <= 7


def test_generated_script_runs_self_test_without_gpu_libraries(tmp_path):
    source = build_kaggle_script.build()
    script_path = tmp_path / "run_clinasr.py"
    script_path.write_text(source, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script_path), "--self-test"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "self-test: OK" in result.stdout
    assert "schema_audit: inspected 2 fixture records" in result.stdout


def test_check_mode_passes_when_output_up_to_date(tmp_path, monkeypatch):
    rebuilt = build_kaggle_script.build()
    fake_output = tmp_path / "run_clinasr.py"
    fake_output.write_text(rebuilt, encoding="utf-8")
    monkeypatch.setattr(build_kaggle_script, "OUTPUT_PATH", fake_output)
    assert build_kaggle_script.main(["--check"]) == 0


def test_check_mode_fails_when_output_stale(tmp_path, monkeypatch):
    fake_output = tmp_path / "run_clinasr.py"
    fake_output.write_text("stale content that does not match a fresh build\n", encoding="utf-8")
    monkeypatch.setattr(build_kaggle_script, "OUTPUT_PATH", fake_output)
    assert build_kaggle_script.main(["--check"]) == 1


def test_module_style_imports_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="module import not allowed"):
        build_kaggle_script.check_flat_imports("x", "from clinasr import stage2\n")
    with pytest.raises(ValueError):
        build_kaggle_script.check_flat_imports("x", "import clinasr.stage2\n")
    build_kaggle_script.check_flat_imports("x", "from clinasr.stage2 import select\n")


def test_bundle_executes_stage_modules_without_name_errors(tmp_path):
    """Import-time execution of the whole bundle plus resolving every
    top-level function's global names catches flat-namespace NameErrors."""
    import ast
    import builtins

    source = build_kaggle_script.build()
    tree = ast.parse(source)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert sorted(loads - defined) == []
