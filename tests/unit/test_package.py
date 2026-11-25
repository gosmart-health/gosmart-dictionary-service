"""Tests for the gosmart_dictionary_service package.

These tests import the installed (editable) package, which is exactly how the
service is consumed at runtime.
"""

import subprocess
import sys
from pathlib import Path

import gosmart_dictionary_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_version_is_exposed() -> None:
    assert gosmart_dictionary_service.__version__ == "0.1.0"


def test_package_lives_under_src_layout() -> None:
    package_dir = Path(gosmart_dictionary_service.__file__).resolve().parent
    assert package_dir == PROJECT_ROOT / "src" / "gosmart_dictionary_service"


def test_module_entry_point_is_importable() -> None:
    """``python -m gosmart_dictionary_service`` must resolve in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", "import gosmart_dictionary_service.__main__ as m; assert callable(m.main)"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, f"Import failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
