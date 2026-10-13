import importlib
import io
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from threading import Event
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import urlopen

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.cache_store import CacheStore


ROOT_DIR = Path(__file__).resolve().parents[1]


class NativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="labelimg native ")
        # Windows runner TEMP can contain an 8.3 alias (e.g. RUNNER~1).
        # API paths are resolved, so fixtures must use the same canonical root.
        cls.root = (Path(cls.temp_dir.name) / "датасет с пробелами").resolve()
        cls.root.mkdir()
        cls.store = CacheStore(cls.root / "cache.sqlite3")
        # Import without creating or modifying the user's project cache.
        with patch("backend.app.cache_store.CacheStore", return_value=cls.store):
            cls.backend = importlib.import_module("backend.app.main")
        cls.cache_patch = patch.object(cls.backend, "CACHE_STORE", cls.store)
        cls.cache_patch.start()
        cls.warmup_patch = patch.object(cls.backend, "schedule_session_fingerprint_warmup")
        cls.warmup_patch.start()
        cls.client = TestClient(cls.backend.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.backend.LOCAL_SESSIONS.clear()
        cls.backend.LOCAL_IMAGE_PATHS.clear()
        cls.warmup_patch.stop()
        cls.cache_patch.stop()
        cls.temp_dir.cleanup()

    def make_image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), "red").save(path)
        return path

    def open_image(self, path):
        response = self.client.post(
            "/api/local/sessions/open-image-path", json={"path": f'"{path}"'}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_native_and_quoted_paths(self):
        path = self.make_image(self.root / "пути" / "картинка.png")
        for raw_path in (str(path), path.as_posix(), f'"{path}"', f"'{path}'"):
            with self.subTest(raw_path=raw_path):
                self.assertEqual(self.backend.resolve_local_request_path(raw_path), path.resolve())
        if os.name == "nt":
            self.assertIsNone(self.backend.translate_windows_path_for_wsl(str(path)))
            self.assertEqual(self.store.dataset_key_for_path(path), self.store.dataset_key_for_path(str(path).upper()))
        else:
            self.assertEqual(
                self.backend.translate_windows_path_for_wsl(r"C:\Users\you\Pictures"),
                "/mnt/c/Users/you/Pictures",
            )

    def test_annotation_roundtrip_and_atomic_overwrite(self):
        for annotation_format, extension in (("yolo", ".txt"), ("voc", ".xml")):
            with self.subTest(annotation_format=annotation_format):
                path = self.make_image(self.root / annotation_format / "фото.png")
                sidecar = path.with_suffix(extension)
                sidecar.write_text("<annotation/>" if extension == ".xml" else "", encoding="utf-8")
                session = self.open_image(path)
                image_id = session["images"][0]["id"]
                session_id = session["sessionId"]
                url = f"/api/local/sessions/{session_id}/annotations/{image_id}"
                for label in ("кот", "собака"):
                    response = self.client.put(url, json={
                        "annotations": [{"label": label, "x": 10, "y": 12, "width": 30, "height": 20}],
                        "projectClasses": ["кот", "собака"],
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["format"], annotation_format)
                    # Reopen to exercise disk/manifest reads, not just the live session.
                    reopened = self.open_image(path)
                    result = self.client.get(
                        f"/api/local/sessions/{reopened['sessionId']}/annotations/{reopened['images'][0]['id']}"
                    )
                    self.assertEqual(result.status_code, 200, result.text)
                    annotation = result.json()["annotations"][0]
                    self.assertEqual(annotation["label"], label)
                    self.assertAlmostEqual(annotation["x"], 10, places=2)
                    self.assertAlmostEqual(annotation["width"], 30, places=2)
                self.assertTrue(sidecar.is_file())
                self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_upload_reserved_name_then_delete(self):
        dataset = self.root / "загрузка"
        dataset.mkdir()
        self.make_image(dataset / "original.png")
        response = self.client.post(
            "/api/local/sessions/open-directory-path", json={"path": str(dataset)}
        )
        self.assertEqual(response.status_code, 200, response.text)
        session_id = response.json()["sessionId"]
        image_bytes = io.BytesIO()
        Image.new("RGB", (10, 10), "blue").save(image_bytes, format="PNG")
        upload_url = f"/api/local/sessions/{session_id}/images/upload"
        for filename in ("CON.png", "COM1.photo.png", "обычное фото.png"):
            with self.subTest(filename=filename):
                response = self.client.post(upload_url, params={"filename": filename}, content=image_bytes.getvalue())
                self.assertEqual(response.status_code, 200, response.text)
                image = response.json()["image"]
                expected_name = f"_{filename}" if filename.startswith(("CON", "COM1")) else filename
                self.assertEqual(image["name"], expected_name)
                self.assertTrue((dataset / expected_name).is_file())
                image_url = f"/api/local/sessions/{session_id}/images/{image['id']}"
                self.assertEqual(self.client.get(image_url).content, image_bytes.getvalue())
                duplicate = self.client.post(upload_url, params={"filename": filename}, content=image_bytes.getvalue())
                self.assertEqual(duplicate.status_code, 409)
                deleted = self.client.delete(image_url)
                self.assertEqual(deleted.status_code, 200, deleted.text)
                self.assertFalse((dataset / expected_name).exists())
        self.assertFalse(list(dataset.glob("*.upload")))

    def test_sam_platform_and_dependency_selection(self):
        for system in ("Windows", "Linux"):
            with patch.object(self.backend.platform, "system", return_value=system):
                self.backend.ensure_supported_sam_runtime_platform()
        with patch.object(self.backend.platform, "system", return_value="Darwin"):
            with self.assertRaises(RuntimeError):
                self.backend.ensure_supported_sam_runtime_platform()
        with patch.object(self.backend, "run_pip_command") as pip, \
             patch.object(self.backend.os, "name", "nt"), \
             patch.object(self.backend, "package_version", return_value="2.9.1+cu126"):
            self.backend.install_sam_runtime_dependencies("cuda", plugin_id="sam-3-1")
            self.assertIn("torch>=2.7,<2.11", pip.call_args_list[0].args[0])
            self.assertIn("triton-windows>=3.5,<3.6", pip.call_args_list[1].args[0])
        with patch.object(self.backend, "run_pip_command") as pip, \
             patch.object(self.backend.os, "name", "posix"):
            self.backend.install_sam_runtime_dependencies("cuda", plugin_id="sam-3-1")
            self.assertEqual(pip.call_count, 2)
        with patch.object(self.backend, "run_pip_command") as pip, \
             patch.object(self.backend, "detect_nvidia_gpu", return_value=False):
            plugin = self.backend.PLUGIN_DEFINITIONS["sam-3-1"]
            self.backend.run_plugin_runtime_install_job(plugin, "auto")
            pip.assert_not_called()

    def test_new_image_preserves_shared_yolo_classes(self):
        dataset = self.root / "shared-classes"
        self.make_image(dataset / "old.png")
        self.make_image(dataset / "new.png")
        (dataset / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        original = "1 0.5 0.5 0.2 0.2\n"
        (dataset / "old.txt").write_text(original, encoding="utf-8")
        session = self.open_image(dataset / "new.png")
        result = self.client.put(
            f"/api/local/sessions/{session['sessionId']}/annotations/{session['images'][0]['id']}",
            json={"annotations": [{"label": "cat", "x": 5, "y": 5, "width": 20, "height": 20}],
                  "projectClasses": ["cat"]},
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual((dataset / "classes.txt").read_text().splitlines(), ["cat", "dog"])
        self.assertEqual((dataset / "old.txt").read_text(), original)
        old = self.open_image(dataset / "old.png")
        result = self.client.get(f"/api/local/sessions/{old['sessionId']}/annotations/{old['images'][0]['id']}")
        self.assertEqual(result.json()["annotations"][0]["label"], "dog")

    def test_directory_cache_detects_new_images_and_sidecars(self):
        dataset = self.root / "external-changes"
        self.make_image(dataset / "a.png")
        url = "/api/local/sessions/open-directory-path"
        first = self.client.post(url, json={"path": str(dataset)}).json()
        self.assertEqual(len(first["images"]), 1)
        self.make_image(dataset / "b.png")
        second = self.client.post(url, json={"path": str(dataset)}).json()
        self.assertEqual(len(second["images"]), 2)
        (dataset / "classes.txt").write_text("cat\n", encoding="utf-8")
        (dataset / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        third = self.client.post(url, json={"path": str(dataset)}).json()
        image = next(i for i in third["images"] if i["name"] == "a.png")
        self.assertEqual(image["annotationCount"], 1)
        annotations = self.client.get(f"/api/local/sessions/{third['sessionId']}/annotations/{image['id']}").json()
        self.assertEqual(annotations["annotations"][0]["label"], "cat")
        (dataset / "a.txt").unlink()
        fourth = self.client.post(url, json={"path": str(dataset)}).json()
        image = next(i for i in fourth["images"] if i["name"] == "a.png")
        self.assertEqual(image["annotationCount"], 0)
        self.assertIsNone(image["annotationFormat"])
        (dataset / "b.png").unlink()
        self.assertEqual(len(self.client.post(url, json={"path": str(dataset)}).json()["images"]), 1)

    def test_class_rename_covers_unloaded_yolo_and_voc_images(self):
        dataset = self.root / "rename-all"
        self.make_image(dataset / "a.png")
        self.make_image(dataset / "b.png")
        self.make_image(dataset / "voc.png")
        (dataset / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        original = "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
        for filename in ("a.txt", "b.txt"):
            (dataset / filename).write_text(original, encoding="utf-8")
        (dataset / "voc.xml").write_text(
            "<annotation><object><name>cat</name><difficult>0</difficult>"
            "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox>"
            "</object></annotation>", encoding="utf-8")
        session = self.client.post("/api/local/sessions/open-directory-path", json={"path": str(dataset)}).json()
        # Do not fetch the second image's annotations before renaming.
        url = f"/api/local/sessions/{session['sessionId']}/classes/rename"
        response = self.client.post(url, json={"sourceLabel": "cat", "targetLabel": "kitten"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((dataset / "classes.txt").read_text().splitlines(), ["kitten", "dog"])
        for filename in ("a.txt", "b.txt"):
            self.assertEqual((dataset / filename).read_text(), original)
        for image in session["images"]:
            response = self.client.get(f"/api/local/sessions/{session['sessionId']}/annotations/{image['id']}")
            labels = [a["label"] for a in response.json()["annotations"]]
            self.assertIn("kitten", labels)
            self.assertNotIn("cat", labels)
        before = (dataset / "classes.txt").read_text()
        conflict = self.client.post(url, json={"sourceLabel": "kitten", "targetLabel": "dog"})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual((dataset / "classes.txt").read_text(), before)

    def test_native_file_picker_routes(self):
        path = self.make_image(self.root / "dialog" / "image.png")
        with patch.object(self.backend, "choose_local_image", return_value=str(path)):
            response = self.client.post("/api/local/sessions/open-image")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["images"][0]["name"], "image.png")
        with patch.object(self.backend, "choose_local_directory", return_value=None):
            self.assertEqual(self.client.post("/api/local/sessions/open-directory").json(), {"cancelled": True})

    def test_class_rename_rolls_back_if_a_write_fails(self):
        self.assert_class_rename_rollback(self.root / "rename-rollback")

    def test_class_rename_rollback_with_noncanonical_path(self):
        # This names the same directory but resolves differently, just like a
        # Windows 8.3 TEMP alias. Exercise it on both CI operating systems.
        alias = self.root.parent / ".." / self.root.parent.name / self.root.name
        self.assertNotEqual(alias, alias.resolve())
        self.assertEqual(alias.resolve(), self.root)
        self.assert_class_rename_rollback(alias / "rename-rollback-alias")

    def assert_class_rename_rollback(self, dataset):
        self.make_image(dataset / "a.png")
        self.make_image(dataset / "b.png")
        classes = dataset / "classes.txt"
        classes.write_text("cat\ndog\n", encoding="utf-8")
        (dataset / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        voc = dataset / "b.xml"
        original_voc = "<annotation><object><name>cat</name></object></annotation>"
        voc.write_text(original_voc, encoding="utf-8")
        original_classes_bytes = classes.read_bytes()
        original_voc_bytes = voc.read_bytes()
        session = self.client.post("/api/local/sessions/open-directory-path", json={"path": str(dataset)}).json()
        real_write = self.backend.write_text_atomically
        attempted_paths = []
        failed_paths = []

        def fail_voc_write(path, contents):
            canonical_path = path.resolve()
            attempted_paths.append(canonical_path)
            if canonical_path == voc.resolve():
                failed_paths.append(canonical_path)
                raise PermissionError("Annotation is read-only")
            real_write(path, contents)

        with patch.object(self.backend, "write_text_atomically", side_effect=fail_voc_write):
            response = self.client.post(
                f"/api/local/sessions/{session['sessionId']}/classes/rename",
                json={"sourceLabel": "cat", "targetLabel": "kitten"},
            )
        self.assertEqual(failed_paths, [voc.resolve()], "The write failure must actually be injected")
        self.assertEqual(response.status_code, 500, response.text)
        # Verify that a successful first write was followed by the failing VOC
        # write and restoration of the class list, including its line endings.
        self.assertEqual(attempted_paths, [classes.resolve(), voc.resolve(), classes.resolve()])
        self.assertEqual(classes.read_bytes(), original_classes_bytes)
        self.assertEqual(voc.read_bytes(), original_voc_bytes)

    def test_unused_yolo_class_can_be_renamed(self):
        dataset = self.root / "rename-unused"
        self.make_image(dataset / "a.png")
        classes = dataset / "classes.txt"
        classes.write_text("cat\ndog\n", encoding="utf-8")
        session = self.client.post("/api/local/sessions/open-directory-path", json={"path": str(dataset)}).json()
        response = self.client.post(
            f"/api/local/sessions/{session['sessionId']}/classes/rename",
            json={"sourceLabel": "cat", "targetLabel": "kitten"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(classes.read_text().splitlines(), ["kitten", "dog"])

    def test_pip_subprocess_from_unicode_directory(self):
        self.backend.run_pip_command(["--version"], cwd=self.root)

    def test_image_metadata_and_bounded_preview(self):
        path = self.root / "preview.png"
        with Image.new("RGBA", (1600, 1200), (255, 0, 0, 128)) as image:
            image.save(path)
        session = self.open_image(path)
        image_id = session["images"][0]["id"]
        base_url = f"/api/local/sessions/{session['sessionId']}/images/{image_id}"
        with patch("PIL.Image.Image.load", side_effect=AssertionError("metadata must not decode pixels")):
            response = self.client.get(f"{base_url}/metadata")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"width": 1600, "height": 1200})
        for url in (f"{base_url}/preview", f"/api/local/images/{image_id}/preview"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(response.content)) as preview:
                self.assertEqual(preview.size, (512, 384))
                self.assertEqual(preview.mode, "RGBA")
        self.assertEqual(self.client.get(f"{base_url}-missing/metadata").status_code, 404)
        self.assertEqual(self.client.get("/api/local/images/missing/preview").status_code, 404)

    def make_duplicate_search_fixture(self, count):
        session = self.backend.create_local_session(self.root, "memory test")
        images = [self.backend.SessionImage(
            id=f"{session.id}-{index}", name=f"{index}.png", relative_path=f"{index}.png",
            full_path=self.root / f"{index}.png", file_size=1, mtime_ns=1,
        ) for index in range(count)]
        session.images = images
        session.images_by_id = {image.id: image for image in images}
        with patch.object(self.backend, "Thread"):
            job = self.backend.start_duplicate_search_job(session, 92, images)
        return session, images, job

    def test_camera_jpeg_metadata_and_preview_match_browser_orientation(self):
        path = self.root / "camera.jpg"
        with Image.new("RGB", (1600, 1200), "red") as image:
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
        session = self.open_image(path)
        image_id = session["images"][0]["id"]
        url = f"/api/local/sessions/{session['sessionId']}/images/{image_id}"
        with patch("PIL.Image.Image.load", side_effect=AssertionError("metadata must not decode pixels")):
            response = self.client.get(f"{url}/metadata")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"width": 1200, "height": 1600})
        response = self.client.get(f"{url}/preview")
        with Image.open(io.BytesIO(response.content)) as preview:
            self.assertEqual(preview.size, (384, 512))
            self.assertIsNone(preview.getexif().get(274))

    def test_cached_directory_job_returns_before_validation_and_reuses_running_job(self):
        path = self.make_image(self.root / "background-open" / "image.png")
        payload = self.open_image(path)
        cached_session = self.backend.LOCAL_SESSIONS[payload["sessionId"]]
        entered = Event()
        release = Event()

        def slow_cache(_root, progress_callback=None):
            progress_callback(phase="validating", processed=0, total=71190)
            entered.set()
            release.wait(5)
            return cached_session

        with patch.object(self.backend, "build_cached_directory_session", side_effect=slow_cache) as cache:
            job = None
            try:
                response = self.client.post("/api/local/sessions/open-directory-path-job", json={"path": str(path.parent)})
                self.assertEqual(response.status_code, 200, response.text)
                job_id = response.json()["jobId"]
                job = self.backend.LOCAL_SESSION_JOBS[job_id]
                self.assertTrue(entered.wait(1))
                self.assertFalse(release.is_set())
                progress = self.client.get(f"/api/local/session-jobs/{job_id}").json()
                self.assertEqual(progress["status"], "running")
                self.assertEqual(progress["phase"], "validating")
                self.assertEqual(progress["totalImages"], 71190)
                repeated = self.client.post("/api/local/sessions/open-directory-path-job", json={"path": str(path.parent)}).json()
                self.assertEqual(repeated["jobId"], job_id)
                self.assertEqual(cache.call_count, 1)
                self.assertEqual(self.client.get("/api/health").status_code, 200)
            finally:
                release.set()
                if job is not None:
                    deadline = time.monotonic() + 5
                    while job.status == "running" and time.monotonic() < deadline:
                        time.sleep(0.01)
            self.assertEqual(job.status, "completed")
            self.assertEqual(job.session_payload["images"][0]["id"], payload["images"][0]["id"])

    def test_cached_directory_job_reports_validation_errors(self):
        with patch.object(self.backend, "build_cached_directory_session", side_effect=PermissionError("cannot read dataset")):
            job = self.backend.start_directory_open_job(self.root / "validation-error")
            deadline = time.monotonic() + 5
            while job.status == "running" and time.monotonic() < deadline:
                time.sleep(0.01)
            response = self.client.get(f"/api/local/session-jobs/{job.id}").json()
            self.assertEqual(response["status"], "failed")
            self.assertIn("cannot read dataset", response["error"])

    def test_directory_scan_does_not_publish_the_full_list_every_24_images(self):
        session, images, _job = self.make_duplicate_search_fixture(500)
        images_iter = iter(images)
        partial_sizes = []

        def on_progress(**progress):
            if progress.get("session") is not None:
                partial_sizes.append(len(progress["session"]["images"]))

        with patch.object(self.backend, "iter_directory_images", return_value=[image.full_path for image in images]), \
             patch.object(self.backend, "build_session_image", side_effect=lambda *_: next(images_iter)), \
             patch.object(self.backend.CACHE_STORE, "save_dataset_manifest"), \
             patch.object(self.backend.time, "monotonic", return_value=0):
            result = self.backend.scan_directory_session(session, progress_callback=on_progress, publish_intermediate=True)
        self.assertEqual(len(result["images"]), 500)
        self.assertEqual(partial_sizes, [1, 4, 12])

    def test_duplicate_search_stops_at_result_limit_and_reports_partial_scan(self):
        session, images, job = self.make_duplicate_search_fixture(300)
        with patch.object(self.backend, "load_or_build_duplicate_image_signature", return_value=({}, False)), \
             patch.object(self.backend, "compare_duplicate_signatures", return_value=(100, "identity")) as compare:
            self.backend.run_duplicate_search_job(job.id, session.root_path, images)
        limit = self.backend.MAX_DUPLICATE_SEARCH_MATCHES
        self.assertEqual(len(job.matches), limit)
        self.assertEqual(compare.call_count, limit)
        self.assertEqual(job.status, "completed")
        self.assertTrue(job.result_limit_reached)
        self.assertLess(job.processed_pairs, job.total_pairs)
        self.assertLess(job.processed_images, job.total_images)
        response = self.client.get(f"/api/local/duplicate-search-jobs/{job.id}").json()
        self.assertTrue(response["resultLimitReached"])
        self.assertEqual(response["matchCount"], limit)
        self.assertEqual(len(response["matches"]), limit)
        unchanged = self.client.get(f"/api/local/duplicate-search-jobs/{job.id}", params={"after_revision": job.revision}).json()
        self.assertIsNone(unchanged["matches"])

    def test_duplicate_scan_can_finish_completely_and_old_results_are_released(self):
        for _ in range(6):
            session, images, job = self.make_duplicate_search_fixture(4)
            with patch.object(self.backend, "load_or_build_duplicate_image_signature", return_value=({}, False)), \
                 patch.object(self.backend, "compare_duplicate_signatures", return_value=(10, "identity")):
                self.backend.run_duplicate_search_job(job.id, session.root_path, images)
            self.assertEqual(job.processed_pairs, 6)
            self.assertEqual(job.processed_images, 4)
            self.assertFalse(job.result_limit_reached)
            self.assertEqual(job.matches, [])
        completed = [job for job in self.backend.DUPLICATE_SEARCH_JOBS.values() if job.status != "running"]
        self.assertLessEqual(len(completed), self.backend.MAX_RETAINED_DUPLICATE_SEARCH_JOBS)


class NativeCacheTests(unittest.TestCase):
    def test_connections_close_and_transactions_commit_or_roll_back(self):
        connections = []
        connect = sqlite3.connect

        def track_connection(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as temp_dir, \
             patch("backend.app.cache_store.sqlite3.connect", side_effect=track_connection):
            db_path = Path(temp_dir) / "cache.sqlite3"
            store = CacheStore(db_path)
            store.merge_app_state({"sidebarVisible": False})
            self.assertFalse(store.load_app_state()["sidebarVisible"])
            with self.assertRaises(RuntimeError):
                with store._connect() as connection:
                    connection.execute("DELETE FROM app_settings")
                    raise RuntimeError("Abort transaction")
            self.assertFalse(store.load_app_state()["sidebarVisible"])
            store.compact_database()
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
            # Kept references must not hold Windows file handles open.
            db_path.unlink()


class NativeLauncherTests(unittest.TestCase):
    def test_server_from_another_working_directory(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="labelimg launch ") as temp_dir, \
             tempfile.TemporaryFile(mode="w+b") as logs:
            process = subprocess.Popen(
                [sys.executable, str(ROOT_DIR / "main.py"), "--port", str(port)],
                cwd=temp_dir, stdout=logs, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        logs.seek(0)
                        self.fail(logs.read().decode("utf-8", errors="replace"))
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                            self.assertIn(b'"status":"ok"', response.read())
                        break
                    except (URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    self.fail("Backend did not start within 20 seconds")
                with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    self.assertIn(b"<!doctype html", response.read().lower())
                with urlopen(f"http://127.0.0.1:{port}/docs", timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
