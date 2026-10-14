import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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
            "if os.environ.get('LABELIMG_TEST_CTRL_C'):\n"
            " import signal, time\n"
            " from pathlib import Path\n"
            " def stopped(*_):\n"
            "  print('SHUTDOWN_COMPLETE', flush=True)\n"
            "  print('SERVER_STDERR', file=sys.stderr, flush=True)\n"
            "  sys.exit(0)\n"
            " signal.signal(signal.SIGINT, stopped)\n"
            " Path(os.environ['LABELIMG_TEST_READY']).touch()\n"
            " while True: time.sleep(.05)\n"
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

    def test_start_ctrl_c_exits_without_waiting_for_console_confirmation(self):
        ready = self.root / "ctrl-c.ready"
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        command = f'cmd.exe /d /s /c ""{self.root / "start.bat"}""'
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                command, cwd=ROOT_DIR, stdin=subprocess.PIPE, stdout=output,
                stderr=subprocess.STDOUT, startupinfo=startup,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                env={**os.environ, "PYTHONUTF8": "1", "LABELIMG_TEST_CTRL_C": "1",
                     "LABELIMG_TEST_READY": str(ready)},
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(ready.exists(), "Launcher did not reach the server")
                # Signal only this test's private console, never the test runner
                # or a user's running server. Keep stdin open to detect a prompt.
                sender = (
                    "import ctypes,sys,time; "
                    "k=ctypes.WinDLL('kernel32',use_last_error=True); "
                    "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
                    "k.SetConsoleCtrlHandler(None,True); "
                    "assert k.GenerateConsoleCtrlEvent(0,0); "
                    "time.sleep(.25); k.FreeConsole()"
                )
                subprocess.run(
                    [sys.executable, "-c", sender, str(process.pid)], check=True,
                    creationflags=subprocess.DETACHED_PROCESS, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                )
                self.assertEqual(process.wait(timeout=5), 0)
                output.seek(0)
                captured = output.read()
                self.assertIn(b"SHUTDOWN_COMPLETE", captured)
                self.assertIn(b"SERVER_STDERR", captured)
                self.assertNotIn(b"[Y", captured)
            finally:
                if process.poll() is None:
                    subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"], capture_output=True)
                    process.wait(timeout=5)
                process.stdin.close()


if __name__ == "__main__":
    unittest.main()
