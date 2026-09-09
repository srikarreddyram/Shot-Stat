"""
Smoke test for r/validate_defender_signal.R.

Running the real analysis needs data/exports/shots_for_r_validation.csv
(a ~180MB export produced by src.analysis.export_for_r_validation, itself a
few minutes of work rebuilding the training feature matrix) — too heavy for
routine test runs. This instead confirms the script is present and
syntactically valid R, which is the failure mode a refactor could actually
introduce without anyone noticing until the next full run.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

RSCRIPT = shutil.which("Rscript")
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "r" / "validate_defender_signal.R"

pytestmark = pytest.mark.skipif(RSCRIPT is None, reason="Rscript not installed")


def test_r_script_is_syntactically_valid():
    result = subprocess.run(
        [RSCRIPT, "-e", f"parse(file='{SCRIPT_PATH}')"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_r_script_exists():
    assert SCRIPT_PATH.exists()
