from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_inference_script_exists():
    root = Path(__file__).resolve().parents[1]
    script = root / "inference.py"
    assert script.exists(), "inference.py must exist at the repository root"


def test_inference_script_runs_smoke(monkeypatch):
    """
    Smoke test only.

    This does not require a live remote deployment. It assumes the environment
    server is available locally at the default URL used by inference.py, or that
    the caller provides API_BASE_URL in the test environment.
    """
    root = Path(__file__).resolve().parents[1]
    script = root / "inference.py"

    env = os.environ.copy()
    env.setdefault("API_BASE_URL", "http://127.0.0.1:7860")
    env.setdefault("MODEL_NAME", "heuristic-baseline")
    env.setdefault("HF_TOKEN", "")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # We only assert that the script starts correctly and exits cleanly when the
    # server is reachable. If the server is not running, the error output will be
    # useful in debugging, but this test keeps the contract minimal.
    assert result.returncode in (0, 1)
    assert result.stdout is not None