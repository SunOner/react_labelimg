"""Persistent video references and progress; original videos are never copied."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from threading import RLock

from fastapi import HTTPException


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".mts", ".m2ts"}
VIDEO_METADATA_LOCK = RLock()
DEFAULT_STATE = {
    "positionSeconds": 0, "durationSeconds": 0, "width": 0, "height": 0,
    "stepFps": 30, "sampleIntervalSeconds": 1, "imageFormat": "png", "playbackRate": 1,
}


def identify_video_path(path: Path):
    """Match frontend identifyVideo without reading a whole multi-gigabyte file."""
    size = path.stat().st_size
    with path.open("rb") as source:
        first = source.read(65536)
        source.seek(max(0, size - 65536))
        last = source.read(65536)
    value = 0xcbf29ce484222325
    for byte in path.name.encode("utf-8") + str(size).encode("ascii") + first + last:
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f"{value:016x}"


class VideoMetadataStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def path_for(self, video_id: str):
        if not re.fullmatch(r"[a-f0-9]{16}", video_id):
            raise HTTPException(status_code=422, detail="Invalid video identifier")
        return self.directory / f"{video_id}.json"

    @staticmethod
    def dataset_key(root: Path):
        return hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()

    def read(self, video_id: str):
        path = self.path_for(video_id)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("id") != video_id or not isinstance(record.get("datasets"), dict):
                raise ValueError("Invalid video metadata")
            return record
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Video metadata not found. Select the original video again.") from exc
        except (ValueError, TypeError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Could not read video metadata: {path.name}") from exc

    def write(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(record["id"])
        fd, name = tempfile.mkstemp(prefix=f".{record['id']}-", suffix=".tmp", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
                json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def dataset(self, record, root: Path):
        key = self.dataset_key(root)
        if key not in record["datasets"]:
            record["datasets"][key] = {"rootPath": str(root.resolve()), "state": dict(DEFAULT_STATE), "frames": {}}
        return record["datasets"][key]

    def public(self, record, root: Path):
        dataset = self.dataset(record, root)
        source_path = record.get("sourcePath")
        return {
            "id": record["id"], "name": record["name"], "sizeBytes": record["sizeBytes"],
            "sourcePath": source_path,
            "sourceMissing": bool(source_path and not Path(source_path).is_file()),
            "state": {**DEFAULT_STATE, **dataset["state"]},
            "frameCount": len(dataset["frames"]), "updatedAt": record["updatedAt"],
        }

    def register(self, video_id: str, name: str, size: int, root: Path, images, source_path: Path | None = None):
        with VIDEO_METADATA_LOCK:
            path = self.path_for(video_id)
            now = datetime.now(timezone.utc).isoformat()
            record = self.read(video_id) if path.exists() else {
                "version": 1, "id": video_id, "name": name, "sizeBytes": size,
                "sourcePath": None, "createdAt": now, "datasets": {},
            }
            if record["name"] != name or record["sizeBytes"] != size:
                raise HTTPException(status_code=409, detail="Video identifier conflicts with an existing source")
            if source_path is not None:
                record["sourcePath"] = str(source_path.resolve())
            dataset = self.dataset(record, root)
            pattern = re.compile(rf"_{video_id}_t(\d+)us\.(png|jpe?g)$", re.IGNORECASE)
            # Recover relationships for frames saved before persistent metadata existed.
            for image in images:
                match = pattern.search(image["name"])
                if match:
                    dataset["frames"][image["relativePath"]] = {
                        "timeSeconds": int(match[1]) / 1_000_000,
                        "imageRelativePath": image["relativePath"],
                        "annotationFormat": image.get("annotationFormat"),
                    }
            record["updatedAt"] = now
            self.write(record)
            return self.public(record, root)

    def list_for_dataset(self, root: Path):
        with VIDEO_METADATA_LOCK:
            videos = []
            for path in self.directory.glob("*.json"):
                if not re.fullmatch(r"[a-f0-9]{16}", path.stem):
                    continue
                record = self.read(path.stem)
                if self.dataset_key(root) in record["datasets"]:
                    videos.append(self.public(record, root))
            return sorted(videos, key=lambda item: item["updatedAt"], reverse=True)

    def save_state(self, video_id: str, root: Path, updates):
        with VIDEO_METADATA_LOCK:
            record = self.read(video_id)
            dataset = self.dataset(record, root)
            state = {**DEFAULT_STATE, **dataset["state"], **updates}
            if state["durationSeconds"] > 0:
                state["positionSeconds"] = min(state["positionSeconds"], state["durationSeconds"])
            dataset["state"] = state
            record["updatedAt"] = datetime.now(timezone.utc).isoformat()
            self.write(record)
            return self.public(record, root)

    def validate_frame(self, video_id: str, root: Path, name: str, timestamp: float):
        record = self.read(video_id)
        if self.dataset_key(root) not in record["datasets"]:
            raise HTTPException(status_code=409, detail="Select this video for the current dataset before saving frames")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > 1_000_000_000:
            raise HTTPException(status_code=422, detail="Invalid video frame timestamp")
        key = math.floor(timestamp * 1_000_000 + 0.5)
        if not re.search(rf"_{video_id}_t0*{key}us\.(png|jpe?g)$", name, re.IGNORECASE):
            raise HTTPException(status_code=422, detail="Frame filename does not match its source video and timestamp")
        return record

    def record_frame(self, video_id: str, root: Path, image, timestamp: float):
        with VIDEO_METADATA_LOCK:
            record = self.validate_frame(video_id, root, image["name"], timestamp)
            dataset = self.dataset(record, root)
            dataset["frames"][image["relativePath"]] = {
                "timeSeconds": timestamp, "imageRelativePath": image["relativePath"],
                "annotationFormat": image.get("annotationFormat"),
            }
            record["updatedAt"] = datetime.now(timezone.utc).isoformat()
            self.write(record)

    def original_path(self, video_id: str):
        record = self.read(video_id)
        source = record.get("sourcePath")
        if not source:
            raise HTTPException(status_code=404, detail="Select the original video again in your browser")
        path = Path(source)
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            raise HTTPException(status_code=404, detail="Original video is unavailable. Saved dataset frames are unaffected.")
        # A different source at the same path must never inherit old frame annotations.
        try:
            source_id = identify_video_path(path)
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Original video is unavailable. Saved dataset frames are unaffected.") from exc
        if source_id != video_id:
            raise HTTPException(status_code=409, detail="The video at this path has changed. Open it again as a new source.")
        return path
