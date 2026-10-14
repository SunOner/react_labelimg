"""Exercise Ctrl+C with real HTTP requests, without touching user datasets."""

import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]

# This fixture runs the real launcher/backend with its cache and dataset isolated.
SERVER_FIXTURE = """
import os, runpy, sys, time
from pathlib import Path
from unittest.mock import patch
root = Path(os.environ['LABELIMG_TEST_ROOT'])
repo = Path(os.environ['LABELIMG_TEST_REPO'])
sys.path.insert(0, str(repo / 'backend'))
from app.cache_store import CacheStore
with patch('app.cache_store.CacheStore', return_value=CacheStore(root / 'cache.sqlite3')):
    from app import main as backend
backend.VIDEO_METADATA_DIR = root / 'video-metadata'
original_save = backend.save_image_annotations_locked
def save(*args, **kwargs):
    (root / 'saving.ready').touch()
    time.sleep(1.2)
    return original_save(*args, **kwargs)
backend.save_image_annotations_locked = save
runpy.run_path(str(repo / 'main.py'), run_name='__main__')
"""


@unittest.skipUnless(os.name == "nt", "Windows console shutdown")
class WindowsServerShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="labelimg shutdown ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        shutil.copyfile(ROOT_DIR / "start.bat", self.root / "start.bat")
        (self.root / "windows.ps1").write_text("exit 0\n", encoding="ascii")
        # Reuse the test runner's dependencies through PYTHONPATH while
        # start.bat runs its real argument and redirection logic.
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(self.root / ".venv")],
            check=True, capture_output=True,
        )
        (self.root / "main.py").write_text(SERVER_FIXTURE, encoding="utf-8")
        from PIL import Image
        self.image = self.root / "dataset" / "image.png"
        self.image.parent.mkdir()
        Image.new("RGB", (100, 80), "red").save(self.image)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        self.logs = tempfile.TemporaryFile()
        self.addCleanup(self.logs.close)
        self.process = subprocess.Popen(
            f'cmd.exe /d /s /c ""{self.root / "start.bat"}" --port {self.port}"',
            cwd=self.root, stdin=subprocess.PIPE, stdout=self.logs,
            stderr=subprocess.STDOUT, startupinfo=startup,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env={**os.environ, "PYTHONUTF8": "1", "LABELIMG_NO_PAUSE": "1",
                 "LABELIMG_TEST_ROOT": str(self.root), "LABELIMG_TEST_REPO": str(ROOT_DIR),
                 "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)},
        )
        self.addCleanup(self.cleanup_server)
        self.clients = []
        self.addCleanup(lambda: [client.close() for client in self.clients])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.assertIsNone(self.process.poll(), self.captured_logs())
            try:
                connection = self.connection()
                connection.request("GET", "/api/health")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                connection.close()
                break
            except OSError:
                time.sleep(.05)
        else:
            self.fail("Server did not start: " + self.captured_logs())
        connection = self.connection()
        connection.request("POST", "/api/local/sessions/open-image-path",
                           json.dumps({"path": str(self.image)}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        session = json.loads(response.read())
        self.base = f'/api/local/sessions/{session["sessionId"]}'
        self.image_id = session["images"][0]["id"]

    def connection(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=8)
        self.clients.append(connection)
        return connection

    def captured_logs(self):
        self.logs.seek(0)
        return self.logs.read().decode("utf-8", errors="replace")

    def cleanup_server(self):
        if self.process.poll() is None:
            # Only the private console/process tree created by this test.
            subprocess.run(["taskkill", "/pid", str(self.process.pid), "/t", "/f"],
                           capture_output=True)
            self.process.wait(timeout=5)
        self.process.stdin.close()

    def stop(self):
        sender = (
            "import ctypes,sys,time; k=ctypes.WinDLL('kernel32',use_last_error=True); "
            "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
            "k.SetConsoleCtrlHandler(None,True); assert k.GenerateConsoleCtrlEvent(0,0); "
            "time.sleep(.25); k.FreeConsole()"
        )
        started = time.monotonic()
        subprocess.run(
            [sys.executable, "-c", sender, str(self.process.pid)], check=True,
            creationflags=subprocess.DETACHED_PROCESS, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        return started

    def assert_finished(self, started, *, limit):
        self.assertEqual(self.process.wait(timeout=8), 0, self.captured_logs())
        self.assertLess(time.monotonic() - started, limit)
        output = self.captured_logs()
        self.assertIn("Application shutdown complete.", output)
        self.assertIn("Finished server process", output)
        self.assertNotIn("[Y", output)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))
        return output

    def test_polling_keepalive_does_not_delay_exit(self):
        connection = self.connection()
        for _ in range(4):
            connection.request("GET", self.base + "/notifications?after_sequence=0")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
        # Keep the browser-like connection open until AFTER the server exits.
        output = self.assert_finished(self.stop(), limit=3)
        self.assertNotIn("timeout graceful shutdown exceeded", output)

    def test_stalled_frame_upload_cannot_hold_launcher_open(self):
        client = socket.create_connection(("127.0.0.1", self.port), timeout=8)
        self.clients.append(client)
        client.sendall((
            f"POST {self.base}/video-frames HTTP/1.1\r\nHost: localhost\r\n"
            "Content-Length: 1000000\r\n"
            "Content-Type: multipart/form-data; boundary=frame\r\n\r\n"
            '--frame\r\nContent-Disposition: form-data; name="file"; filename="frame.png"\r\n'
            "Content-Type: image/png\r\n\r\npartial-image"
        ).encode("ascii"))
        time.sleep(.2)
        output = self.assert_finished(self.stop(), limit=8)
        self.assertIn("timeout graceful shutdown exceeded", output)
        self.assertEqual(list(self.image.parent.glob("*.png")), [self.image])

    def test_annotation_save_finishes_before_exit(self):
        connection = self.connection()
        connection.request(
            "PUT", f"{self.base}/annotations/{self.image_id}",
            json.dumps({"projectClasses": ["player"], "annotations": [
                {"id": "box", "label": "player", "x": 10, "y": 10,
                 "width": 40, "height": 40}]}),
            {"Content-Type": "application/json"},
        )
        deadline = time.monotonic() + 4
        while not (self.root / "saving.ready").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue((self.root / "saving.ready").exists(), self.captured_logs())
        started = self.stop()
        response = connection.getresponse()
        self.assertEqual(response.status, 200, response.read())
        response.read()
        output = self.assert_finished(started, limit=4)
        self.assertNotIn("timeout graceful shutdown exceeded", output)
        sidecars = list(self.image.parent.rglob("image.txt"))
        self.assertEqual(len(sidecars), 1)
        self.assertEqual([float(value) for value in sidecars[0].read_text().split()],
                         [0, .3, .375, .4, .5])

    def test_unread_video_response_cannot_hold_launcher_open(self):
        original = self.root / "original.mp4"
        with original.open("wb") as video:
            video.truncate(16 * 1024 * 1024)
        connection = self.connection()
        connection.request("POST", self.base + "/videos/open-path",
                           json.dumps({"path": str(original)}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        source = json.loads(response.read())
        client = socket.socket()
        self.clients.append(client)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client.settimeout(8)
        client.connect(("127.0.0.1", self.port))
        client.sendall((
            f'GET {self.base}/videos/{source["id"]}/file HTTP/1.1\r\n'
            'Host: localhost\r\n\r\n'
        ).encode("ascii"))
        self.assertIn(b"200 OK", client.recv(4096))
        # A paused/unresponsive browser leaves the large response unread.
        time.sleep(.2)
        output = self.assert_finished(self.stop(), limit=8)
        self.assertIn("timeout graceful shutdown exceeded", output)
        self.assertEqual(original.stat().st_size, 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
