# model_runner.py
#
# Thin wrapper around the marsh_cli binary.
# Runs one YAML config file and returns the path to the output NetCDF.
#
# Usage:
#   nc_path = run_model(yaml_path, cli_binary="marsh_cli")
#
# The binary must be on PATH or an absolute path must be supplied.

from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def run_model(
    yaml_path: str,
    cli_binary: str = "marsh_cli",
    silent: bool = True,
    timeout_s: int = 3600,
) -> str:
    """Run marsh_cli on yaml_path and return the output NetCDF path.

    Parameters
    ----------
    yaml_path:
        Path to the run configuration YAML file.
    cli_binary:
        Name or absolute path of the marsh_cli executable.
    silent:
        If True, pass --silent to suppress progress output.
    timeout_s:
        Maximum wall-clock seconds before the subprocess is killed.

    Returns
    -------
    Path to the NetCDF output file (derived from yaml_path by replacing
    '.yaml' with '.nc', matching what yaml_writer.write_config() sets).

    Raises
    ------
    RuntimeError if the process exits with a non-zero status.
    subprocess.TimeoutExpired if the run exceeds timeout_s.
    """
    cmd = [cli_binary, yaml_path]
    if silent:
        cmd.append("--silent")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"marsh_cli failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    nc_path = str(Path(yaml_path).with_suffix(".nc"))
    return nc_path
