"""Install/verify the optional SAM runtime in the current Python environment."""

import argparse
import ctypes
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen


def windows_cpp_runtime_available():
    try:
        for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
            ctypes.WinDLL(name)
    except OSError:
        return False
    return True


def ensure_windows_cpp_runtime():
    if os.name != "nt" or windows_cpp_runtime_available():
        return
    print("[SAM] Installing Microsoft Visual C++ runtime.", flush=True)
    with tempfile.TemporaryDirectory(prefix="labelimg-vcredist-") as temp_dir:
        installer = Path(temp_dir) / "vc_redist.x64.exe"
        with (
            urlopen("https://aka.ms/vc14/vc_redist.x64.exe", timeout=60) as source,
            installer.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        command = """
$ErrorActionPreference = 'Stop'
$signature = Get-AuthenticodeSignature -LiteralPath $env:LABELIMG_VC_INSTALLER
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
    throw 'Invalid Microsoft Visual C++ installer signature.'
}
$installerProcess = Start-Process -FilePath $env:LABELIMG_VC_INSTALLER -ArgumentList '/install', '/quiet', '/norestart' -WindowStyle Hidden -Wait -PassThru
exit $installerProcess.ExitCode
"""
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            env={**os.environ, "LABELIMG_VC_INSTALLER": str(installer)},
        )
        if result.returncode not in (0, 3010, 1638):
            raise RuntimeError(
                f"Visual C++ runtime installation failed ({result.returncode})."
            )
    if not windows_cpp_runtime_available():
        raise RuntimeError("Visual C++ runtime is unavailable. Restart Windows and retry.")


def ensure_runtime(backend, *, if_available=False, force=False):
    backend.ensure_supported_sam_runtime_host()
    backend.ensure_supported_sam_runtime_platform()
    if not backend.detect_nvidia_gpu():
        if if_available:
            print("[SAM] NVIDIA GPU not detected; manual annotation is available.")
            return
        raise RuntimeError("SAM requires a compatible NVIDIA GPU and driver.")

    ensure_windows_cpp_runtime()
    if not force:
        check = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--check"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if check.returncode == 0:
            backend.store_plugin_runtime_state(
                "sam-3-1", status="ready", device="cuda", message=None,
            )
            print("[SAM] Runtime and CUDA already verified; skipping installation.")
            return

    plugin = backend.PLUGIN_DEFINITIONS["sam-3-1"]
    job = backend.PluginRuntimeInstallJob(
        plugin_id=plugin.id, status="running", requested_profile="cuda",
    )
    backend.PLUGIN_RUNTIME_INSTALL_JOBS[plugin.id] = job
    backend.run_plugin_runtime_install_job(plugin, "cuda")
    if job.status != "completed":
        raise RuntimeError(job.error or "SAM runtime installation failed.")
    print("[SAM] Dependencies installed and CUDA verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--if-available", action="store_true",
        help="Skip SAM on computers without NVIDIA GPUs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reinstall even if the runtime passes verification",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify imports and CUDA without installing anything",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        from backend.app import main as backend

        if args.check:
            backend.resolve_sam_runtime_environment()
            print("[SAM] Runtime and CUDA verified.")
        else:
            ensure_runtime(backend, if_available=args.if_available, force=args.force)
    except (RuntimeError, ImportError, OSError) as exc:
        print(f"[SAM] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
