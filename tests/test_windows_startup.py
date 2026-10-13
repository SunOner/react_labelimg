import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from install_sam import ensure_runtime


ROOT_DIR = Path(__file__).resolve().parents[1]


class SamStartupTests(unittest.TestCase):
    def setUp(self):
        cpp_runtime = patch("install_sam.ensure_windows_cpp_runtime")
        cpp_runtime.start()
        self.addCleanup(cpp_runtime.stop)

    def make_backend(self):
        backend = Mock()
        backend.detect_nvidia_gpu.return_value = True
        backend.PLUGIN_DEFINITIONS = {"sam-3-1": SimpleNamespace(id="sam-3-1")}
        backend.PLUGIN_RUNTIME_INSTALL_JOBS = {}
        backend.PluginRuntimeInstallJob.side_effect = lambda **kwargs: SimpleNamespace(**kwargs, error=None)
        return backend

    def test_existing_runtime_skips_installation(self):
        backend = self.make_backend()
        with patch("install_sam.subprocess.run", return_value=SimpleNamespace(returncode=0)) as check:
            ensure_runtime(backend)
        self.assertIn("--check", check.call_args.args[0])
        backend.run_plugin_runtime_install_job.assert_not_called()

    def test_missing_runtime_installs_and_reports_failure(self):
        for status in ("completed", "failed"):
            with self.subTest(status=status):
                backend = self.make_backend()

                def install(plugin, profile):
                    job = backend.PLUGIN_RUNTIME_INSTALL_JOBS[plugin.id]
                    job.status = status
                    job.error = "Installation failed" if status == "failed" else None

                backend.run_plugin_runtime_install_job.side_effect = install
                with patch("install_sam.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                    if status == "failed":
                        with self.assertRaisesRegex(RuntimeError, "Installation failed"):
                            ensure_runtime(backend)
                    else:
                        ensure_runtime(backend)
                backend.run_plugin_runtime_install_job.assert_called_once()

    def test_gpu_absent_skips_only_when_optional(self):
        backend = self.make_backend()
        backend.detect_nvidia_gpu.return_value = False
        ensure_runtime(backend, if_available=True)
        with self.assertRaisesRegex(RuntimeError, "NVIDIA"):
            ensure_runtime(backend)
        backend.run_plugin_runtime_install_job.assert_not_called()

    def test_force_reinstalls_without_readiness_probe(self):
        backend = self.make_backend()

        def install(plugin, profile):
            backend.PLUGIN_RUNTIME_INSTALL_JOBS[plugin.id].status = "completed"

        backend.run_plugin_runtime_install_job.side_effect = install
        with patch("install_sam.subprocess.run") as check:
            ensure_runtime(backend, force=True)
        check.assert_not_called()
        backend.run_plugin_runtime_install_job.assert_called_once()


@unittest.skipUnless(os.name == "nt", "Windows batch files")
class WindowsBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="labelimg batch ")
        cls.root = Path(cls.temp_dir.name) / "проект с пробелами & !"
        cls.root.mkdir()
        for filename in ("start.bat", "setup.bat", "install_sam.bat"):
            shutil.copyfile(ROOT_DIR / filename, cls.root / filename)
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(cls.root / ".venv")],
            check=True, capture_output=True,
        )
        (cls.root / "windows.ps1").write_text(
            "param([switch]$Rebuild, [switch]$SamOnly, [switch]$RequireSam)\n"
            "Write-Host ('setup: ' + $Rebuild + ' ' + $SamOnly + ' ' + $RequireSam)\n"
            "if ($env:LABELIMG_TEST_SETUP_FAILURE) { exit 9 }\n",
            encoding="ascii",
        )
        (cls.root / "main.py").write_text(
            "import os, sys\nprint(os.getcwd())\nprint(repr(sys.argv[1:]))\n"
            "sys.exit(7 if '--fail' in sys.argv else 0)\n",
            encoding="ascii",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def run_batch(self, filename, arguments="", **env):
        # The outer pair of quotes is required by cmd /s /c for a quoted command.
        command = f'""{self.root / filename}" {arguments}"'
        return subprocess.run(
            "cmd.exe /d /s /c " + command, cwd=ROOT_DIR,
            env={**os.environ, "LABELIMG_NO_PAUSE": "1", "PYTHONUTF8": "1", **env},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_start_forwards_arguments_and_uses_project_directory(self):
        result = self.run_batch("start.bat", '--port 8080 --host "127.0.0.1"')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.root), result.stdout)
        self.assertIn("['--port', '8080', '--host', '127.0.0.1']", result.stdout)

    def test_batch_errors_keep_the_original_exit_code(self):
        self.assertEqual(self.run_batch("start.bat", "--fail").returncode, 7)
        result = self.run_batch("start.bat", LABELIMG_TEST_SETUP_FAILURE="1")
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertNotIn("['", result.stdout)

    def test_setup_and_sam_entry_points(self):
        for filename, expected in (("setup.bat", "setup: True False False"),
                                   ("install_sam.bat", "setup: False True True")):
            with self.subTest(filename=filename):
                result = self.run_batch(filename)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
