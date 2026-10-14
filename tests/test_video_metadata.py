import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.cache_store import CacheStore
from backend.app.video_metadata import VideoMetadataStore, identify_video_path


class VideoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="labelimg video metadata ")
        self.root = Path(self.temporary.name).resolve()
        self.cache = CacheStore(self.root / "cache.sqlite3")
        with patch("backend.app.cache_store.CacheStore", return_value=self.cache):
            self.backend = importlib.import_module("backend.app.main")
        self.patches = [
            patch.object(self.backend, "CACHE_STORE", self.cache),
            patch.object(self.backend, "VIDEO_METADATA_DIR", self.root / "program" / "data" / "video-metadata"),
            patch.object(self.backend, "schedule_session_fingerprint_warmup"),
        ]
        for item in self.patches:
            item.start()
        self.client = TestClient(self.backend.app)
        self.dataset = self.root / "dataset"
        self.session_id = self.open_dataset(self.dataset)
        self.original = self.root / "original videos" / "ролик.mp4"
        self.original.parent.mkdir()
        self.content = b"0123456789-fake-video-source"
        self.original.write_bytes(self.content)
        self.base = f"/api/local/sessions/{self.session_id}"

    def tearDown(self):
        self.client.close()
        self.backend.LOCAL_SESSIONS.clear()
        self.backend.LOCAL_IMAGE_PATHS.clear()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def open_dataset(self, path):
        path.mkdir()
        Image.new("RGB", (100, 80), "red").save(path / "original.png")
        response = self.client.post("/api/local/sessions/open-directory-path", json={"path": str(path)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["sessionId"]

    def open_video(self):
        response = self.client.post(self.base + "/videos/open-path", json={"path": str(self.original)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def save_frame(self, video, timestamp=1):
        image = io.BytesIO()
        Image.new("RGB", (100, 80), "blue").save(image, format="PNG")
        name = f"clip_{video['id']}_t{round(timestamp * 1_000_000):012}us.png"
        return self.client.post(self.base + "/video-frames", files={"file": (name, image.getvalue(), "image/png")}, data={
            "annotations": json.dumps({"annotations": [{"label": "cat", "x": 10, "y": 12, "width": 30, "height": 20}], "projectClasses": ["existing", "cat"]}),
            "video_id": video["id"], "time_seconds": str(timestamp),
        })

    def test_original_stays_in_place_and_metadata_is_only_in_program(self):
        original_stat = self.original.stat()
        video = self.open_video()
        self.assertEqual(video["sourcePath"], str(self.original))
        self.assertEqual(video["id"], identify_video_path(self.original))
        self.assertEqual(self.original.read_bytes(), self.content)
        self.assertEqual(self.original.stat().st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual([path.name for path in self.dataset.iterdir()], ["original.png"])
        self.assertFalse((self.dataset / "Videos").exists())
        metadata = self.backend.VIDEO_METADATA_DIR / f"{video['id']}.json"
        self.assertTrue(metadata.exists())
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["sourcePath"], str(self.original))
        response = self.client.get(self.base + f"/videos/{video['id']}/file", headers={"Range": "bytes=4-10"})
        self.assertEqual(response.status_code, 206, response.text)
        self.assertEqual(response.content, self.content[4:11])

    def test_import_during_fresh_and_cached_loading_survives_later_snapshots(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                dataset = self.root / ("cached loading" if cached else "fresh loading")
                dataset.mkdir()
                for index in range(14):
                    Image.new("RGB", (100, 80), "red").save(dataset / f"original{index:02}.png")
                if cached:
                    self.backend.build_directory_session(dataset)
                saved = None
                snapshots = []

                def progress(**update):
                    nonlocal saved
                    snapshot = update.get("session")
                    if not snapshot:
                        return
                    snapshots.append(snapshot)
                    if saved is None:
                        self.session_id = snapshot["sessionId"]
                        self.base = f"/api/local/sessions/{self.session_id}"
                        response = self.save_frame(self.open_video())
                        self.assertEqual(response.status_code, 200, response.text)
                        saved = response.json()["image"]

                final = self.backend.build_directory_session(dataset, progress_callback=progress)
                self.assertIsNotNone(saved)
                self.assertGreater(len(snapshots), 1)
                for snapshot in snapshots[1:] + [final]:
                    frames = [image for image in snapshot["images"] if image["id"] == saved["id"]]
                    self.assertEqual(len(frames), 1)
                    self.assertEqual(frames[0]["annotationCount"], 1)
                self.assertEqual(len(final["images"]), 15)
                manifest = self.cache.load_dataset_manifest(dataset)
                self.assertEqual(len(manifest["images"]), 15)
                self.assertIn(saved["id"], {image["id"] for image in manifest["images"]})
                response = self.client.get(self.base + f"/annotations/{saved['id']}")
                self.assertEqual(response.json()["annotations"][0]["label"], "cat")

    def test_progress_and_settings_resume_independently_for_each_dataset(self):
        video = self.open_video()
        url = self.base + f"/videos/{video['id']}/state"
        response = self.client.patch(url, json={"positionSeconds": 12.5, "durationSeconds": 60, "width": 1920, "height": 1080, "stepFps": 25, "sampleIntervalSeconds": 1.5, "imageFormat": "jpg", "playbackRate": .5})
        self.assertEqual(response.status_code, 200, response.text)
        restored = self.open_video()
        self.assertEqual(restored["state"]["positionSeconds"], 12.5)
        self.assertEqual(restored["state"]["stepFps"], 25)
        self.assertEqual(restored["state"]["imageFormat"], "jpg")
        other_session = self.open_dataset(self.root / "other dataset")
        other = self.client.post(f"/api/local/sessions/{other_session}/videos/open-path", json={"path": str(self.original)})
        self.assertEqual(other.json()["state"]["positionSeconds"], 0)
        history = self.client.get(self.base + "/videos").json()
        self.assertEqual(history[0]["state"]["positionSeconds"], 12.5)
        self.assertEqual(self.client.post(url, json={"positionSeconds": 999}).json()["state"]["positionSeconds"], 60)

    def test_deleting_original_keeps_frames_labels_and_metadata(self):
        video = self.open_video()
        response = self.save_frame(video)
        self.assertEqual(response.status_code, 200, response.text)
        entry = response.json()["image"]
        image_path = self.dataset / entry["relativePath"]
        annotation_path = self.backend.LOCAL_SESSIONS[self.session_id].images_by_id[entry["id"]].annotation_path
        self.original.unlink()
        missing = self.client.get(self.base + f"/videos/{video['id']}/file")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("unaffected", missing.json()["detail"])
        history = self.client.get(self.base + "/videos").json()
        self.assertTrue(history[0]["sourceMissing"])
        self.assertEqual(history[0]["frameCount"], 1)
        self.assertTrue(image_path.is_file())
        self.assertTrue(annotation_path.is_file())
        annotations = self.client.get(self.base + f"/annotations/{entry['id']}").json()
        self.assertEqual(annotations["annotations"][0]["label"], "cat")
        record = VideoMetadataStore(self.backend.VIDEO_METADATA_DIR).read(video["id"])
        dataset = next(iter(record["datasets"].values()))
        self.assertEqual(dataset["frames"][entry["relativePath"]]["timeSeconds"], 1)

    def test_browser_registration_uploads_no_video_and_recovers_legacy_frame_links(self):
        video_id = "0123456789abcdef"
        legacy = self.dataset / f"clip_{video_id}_t000001000000us.png"
        Image.new("RGB", (100, 80)).save(legacy)
        session = self.client.post("/api/local/sessions/open-directory-path", json={"path": str(self.dataset)}).json()
        url = f"/api/local/sessions/{session['sessionId']}/videos/register"
        response = self.client.post(url, json={"id": video_id, "name": "clip.mp4", "sizeBytes": 5000000000})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["sourcePath"])
        self.assertEqual(response.json()["frameCount"], 1)
        self.assertFalse(list(self.dataset.glob("**/*.mp4")))
        self.assertFalse(list(self.backend.VIDEO_METADATA_DIR.glob("*.tmp")))

    def test_metadata_write_failure_rolls_back_new_frame_and_class_changes(self):
        video = self.open_video()
        classes_path = self.dataset / "classes.txt"
        classes_path.write_bytes(b"existing\r\n")
        (self.dataset / "original.txt").write_bytes(b"")
        original_metadata = (self.backend.VIDEO_METADATA_DIR / f"{video['id']}.json").read_bytes()
        with patch.object(VideoMetadataStore, "write", side_effect=OSError("metadata disk full")):
            with self.assertRaisesRegex(OSError, "metadata disk full"):
                self.save_frame(video)
        self.assertEqual(classes_path.read_bytes(), b"existing\r\n")
        self.assertEqual((self.backend.VIDEO_METADATA_DIR / f"{video['id']}.json").read_bytes(), original_metadata)
        self.assertFalse(list(self.dataset.glob("clip*")))
        self.assertEqual(len(self.backend.LOCAL_SESSIONS[self.session_id].images), 1)

    def test_changed_source_invalid_metadata_and_bad_frame_timestamp_are_rejected(self):
        video = self.open_video()
        self.original.write_bytes(b"a different video")
        response = self.client.get(self.base + f"/videos/{video['id']}/file")
        self.assertEqual(response.status_code, 409)
        response = self.client.post(self.base + "/videos/register", json={"id": "../escape", "name": "x.mp4", "sizeBytes": 1})
        self.assertEqual(response.status_code, 422)
        response = self.client.patch(self.base + f"/videos/{video['id']}/state", json={"stepFps": 0})
        self.assertEqual(response.status_code, 422)
        image = io.BytesIO()
        Image.new("RGB", (100, 80)).save(image, format="PNG")
        response = self.client.post(self.base + "/video-frames", files={"file": ("unrelated.png", image.getvalue())}, data={"annotations": '{"annotations":[]}', "video_id": video["id"], "time_seconds": "1"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.dataset / "unrelated.png").exists())


if __name__ == "__main__":
    unittest.main()
