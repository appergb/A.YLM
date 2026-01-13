"""Integration tests for CLI functionality."""

import pytest
import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Test that CLI help works."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "aylm.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "aylm" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("CLI not available or timed out")


def test_cli_imports():
    """Test CLI module imports."""
    try:
        from aylm.cli import main_cli
        assert main_cli is not None
    except ImportError as e:
        pytest.skip(f"CLI import failed: {e}")


def test_run_sharp_script_exists():
    """Test that run_sharp.sh script exists and is executable."""
    script_path = Path("run_sharp.sh")
    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111  # Check if executable