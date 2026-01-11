"""Pytest configuration for kernel validation tests."""

import subprocess

import pytest
import torch


def _detect_backend(bin_path: str = "./cmhsa.out") -> str:
    """Run the binary with minimal args to detect backend from output."""
    try:
        output = subprocess.check_output(
            [
                bin_path,
                "--batch",
                "1",
                "--n_heads",
                "1",
                "--seq_len",
                "1",
                "--head_dim",
                "16",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        if "CUDA" in output:
            return "cuda"
        return "cpu"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "cpu"


@pytest.fixture(scope="session")
def backend():
    """Auto-detect and return the backend from the compiled binary."""
    return _detect_backend()


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests if CUDA binary detected but no GPU available."""
    backend = _detect_backend()
    if backend == "cuda" and not torch.cuda.is_available():
        skip_marker = pytest.mark.skip(reason="CUDA binary but no GPU available")
        for item in items:
            item.add_marker(skip_marker)
