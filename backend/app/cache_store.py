import copy
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Sequence


DEFAULT_APP_STATE = {
    "sidebarVisible": True,
    "recentDatasets": [],
    "sessionState": None,
    "hotkeys": None,
}


class CacheStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        with self._lock:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS dataset_manifests (
                        dataset_key TEXT PRIMARY KEY,
                        root_path TEXT NOT NULL UNIQUE,
                        label TEXT NOT NULL,
                        image_count INTEGER NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS dataset_images (
                        dataset_key TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        image_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        full_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        annotation_path TEXT,
                        annotation_format TEXT,
                        annotation_count INTEGER NOT NULL,
                        PRIMARY KEY (dataset_key, relative_path),
                        FOREIGN KEY (dataset_key) REFERENCES dataset_manifests(dataset_key) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_dataset_images_dataset_sort
                    ON dataset_images(dataset_key, sort_index);

                    CREATE TABLE IF NOT EXISTS annotation_cache (
                        annotation_key TEXT PRIMARY KEY,
                        annotation_path TEXT NOT NULL,
                        annotation_format TEXT NOT NULL,
                        annotation_mtime_ns INTEGER NOT NULL,
                        image_path TEXT NOT NULL,
                        image_mtime_ns INTEGER NOT NULL,
                        class_source_path TEXT,
                        class_source_mtime_ns INTEGER,
                        payload_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """
                )

    def dataset_key_for_path(self, root_path: Path):
        normalized_root = self._normalize_path(root_path)
        return hashlib.sha1(normalized_root.encode("utf-8")).hexdigest()

    def image_id_for_path(
        self,
        root_path: Path,
        relative_path: str,
        *,
        mtime_ns: int,
        file_size: int,
    ):
        source = (
            f"{self.dataset_key_for_path(root_path)}|{relative_path}|"
            f"{mtime_ns}|{file_size}"
        )
        return hashlib.sha1(source.encode("utf-8")).hexdigest()

    def load_app_state(self):
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT key, value_json FROM app_settings"
                ).fetchall()

        state = copy.deepcopy(DEFAULT_APP_STATE)
        for row in rows:
            try:
                state[row["key"]] = json.loads(row["value_json"])
            except Exception:
                continue
        return state

    def merge_app_state(self, patch: dict[str, Any]):
        state = self.load_app_state()
        state.update(patch)
        now = time.time()

        with self._lock:
            with self._connect() as connection:
                for key, value in state.items():
                    connection.execute(
                        """
                        INSERT INTO app_settings (key, value_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value_json = excluded.value_json,
                            updated_at = excluded.updated_at
                        """,
                        (key, json.dumps(value, ensure_ascii=False), now),
                    )

        return state

    def load_dataset_manifest(self, root_path: Path):
        dataset_key = self.dataset_key_for_path(root_path)

        with self._lock:
            with self._connect() as connection:
                manifest_row = connection.execute(
                    """
                    SELECT label, image_count
                    FROM dataset_manifests
                    WHERE dataset_key = ?
                    """,
                    (dataset_key,),
                ).fetchone()
                if manifest_row is None:
                    return None

                image_rows = connection.execute(
                    """
                    SELECT
                        image_id,
                        name,
                        relative_path,
                        full_path,
                        file_size,
                        mtime_ns,
                        annotation_path,
                        annotation_format,
                        annotation_count
                    FROM dataset_images
                    WHERE dataset_key = ?
                    ORDER BY sort_index
                    """,
                    (dataset_key,),
                ).fetchall()

        return {
            "datasetKey": dataset_key,
            "label": manifest_row["label"],
            "imageCount": manifest_row["image_count"],
            "images": [
                {
                    "id": row["image_id"],
                    "name": row["name"],
                    "relative_path": row["relative_path"],
                    "full_path": row["full_path"],
                    "file_size": row["file_size"],
                    "mtime_ns": row["mtime_ns"],
                    "annotation_path": row["annotation_path"],
                    "annotation_format": row["annotation_format"],
                    "annotation_count": row["annotation_count"],
                }
                for row in image_rows
            ],
        }

    def save_dataset_manifest(
        self,
        root_path: Path,
        session_label: str,
        images: Sequence[dict[str, Any]],
    ):
        dataset_key = self.dataset_key_for_path(root_path)
        now = time.time()
        image_rows = [
            (
                dataset_key,
                index,
                image["id"],
                image["name"],
                image["relative_path"],
                image["full_path"],
                image["file_size"],
                image["mtime_ns"],
                image.get("annotation_path"),
                image.get("annotation_format"),
                image.get("annotation_count", 0),
            )
            for index, image in enumerate(images)
        ]

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO dataset_manifests (
                        dataset_key,
                        root_path,
                        label,
                        image_count,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_key) DO UPDATE SET
                        root_path = excluded.root_path,
                        label = excluded.label,
                        image_count = excluded.image_count,
                        updated_at = excluded.updated_at
                    """,
                    (
                        dataset_key,
                        str(root_path),
                        session_label,
                        len(images),
                        now,
                    ),
                )
                connection.execute(
                    "DELETE FROM dataset_images WHERE dataset_key = ?",
                    (dataset_key,),
                )
                connection.executemany(
                    """
                    INSERT INTO dataset_images (
                        dataset_key,
                        sort_index,
                        image_id,
                        name,
                        relative_path,
                        full_path,
                        file_size,
                        mtime_ns,
                        annotation_path,
                        annotation_format,
                        annotation_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    image_rows,
                )

    def update_dataset_annotation_metadata(
        self,
        root_path: Path,
        relative_path: str,
        annotation_count: int,
        annotation_format: str | None,
    ):
        dataset_key = self.dataset_key_for_path(root_path)

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE dataset_images
                    SET annotation_count = ?, annotation_format = ?
                    WHERE dataset_key = ? AND relative_path = ?
                    """,
                    (
                        annotation_count,
                        annotation_format,
                        dataset_key,
                        relative_path,
                    ),
                )

    def load_annotation_cache(
        self,
        *,
        annotation_path: Path,
        annotation_format: str,
        annotation_mtime_ns: int,
        image_path: Path,
        image_mtime_ns: int,
        class_source_path: Path | None = None,
        class_source_mtime_ns: int | None = None,
    ):
        annotation_key = self._annotation_key_for_path(annotation_path)

        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        annotation_mtime_ns,
                        image_path,
                        image_mtime_ns,
                        class_source_path,
                        class_source_mtime_ns,
                        payload_json
                    FROM annotation_cache
                    WHERE annotation_key = ?
                    """,
                    (annotation_key,),
                ).fetchone()

        if row is None:
            return None

        if row["annotation_mtime_ns"] != annotation_mtime_ns:
            return None
        if row["image_path"] != str(image_path):
            return None
        if row["image_mtime_ns"] != image_mtime_ns:
            return None
        if row["class_source_path"] != self._optional_path_text(class_source_path):
            return None
        if row["class_source_mtime_ns"] != class_source_mtime_ns:
            return None

        try:
            return json.loads(row["payload_json"])
        except Exception:
            return None

    def save_annotation_cache(
        self,
        *,
        annotation_path: Path,
        annotation_format: str,
        annotation_mtime_ns: int,
        image_path: Path,
        image_mtime_ns: int,
        payload: Sequence[dict[str, Any]],
        class_source_path: Path | None = None,
        class_source_mtime_ns: int | None = None,
    ):
        annotation_key = self._annotation_key_for_path(annotation_path)
        now = time.time()

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO annotation_cache (
                        annotation_key,
                        annotation_path,
                        annotation_format,
                        annotation_mtime_ns,
                        image_path,
                        image_mtime_ns,
                        class_source_path,
                        class_source_mtime_ns,
                        payload_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(annotation_key) DO UPDATE SET
                        annotation_path = excluded.annotation_path,
                        annotation_format = excluded.annotation_format,
                        annotation_mtime_ns = excluded.annotation_mtime_ns,
                        image_path = excluded.image_path,
                        image_mtime_ns = excluded.image_mtime_ns,
                        class_source_path = excluded.class_source_path,
                        class_source_mtime_ns = excluded.class_source_mtime_ns,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        annotation_key,
                        str(annotation_path),
                        annotation_format,
                        annotation_mtime_ns,
                        str(image_path),
                        image_mtime_ns,
                        self._optional_path_text(class_source_path),
                        class_source_mtime_ns,
                        json.dumps(list(payload), ensure_ascii=False),
                        now,
                    ),
                )

    def _annotation_key_for_path(self, annotation_path: Path):
        normalized_path = self._normalize_path(annotation_path)
        return hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()

    def _normalize_path(self, path: Path | str):
        normalized_path = str(Path(path).expanduser().resolve())
        if os.name == "nt":
            return normalized_path.lower()
        return normalized_path

    @staticmethod
    def _optional_path_text(path: Path | None):
        return str(path) if path is not None else None
