"""
modules/code_runner.py
Executes generated Python code in a subprocess (isolated from the
main app process) and captures stdout/stderr.

NOTE: This is a *basic* sandbox suitable for a coursework project —
it isolates via subprocess + timeout, but does not fully sandbox
filesystem/network access. Don't expose this publicly without
hardening (e.g. Docker, restricted user, no network).
"""

import subprocess
import sys
import tempfile
import os


def run_code(code: str, timeout: int = 10) -> dict:
    """
    Runs the given Python code string and returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "timed_out": bool
        }
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "timed_out": True,
        }
    finally:
        os.remove(temp_path)


def extract_code_block(markdown_text: str) -> str:
    """
    Extracts the first ```python ... ``` (or ``` ... ```) block from
    an LLM's Markdown response, since handle_generate() returns
    Markdown-wrapped code, not raw code.
    """
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", markdown_text, re.DOTALL)
    return match.group(1).strip() if match else markdown_text.strip()