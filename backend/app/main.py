import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Literal
from uuid import uuid4
from xml.etree import ElementTree

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .cache_store import CacheStore

APP_VERSION = "0.1.0"
ROOT_DIR = Path(__file__).resolve().parents[2]
PREDEFINED_CLASSES_FILE = ROOT_DIR / "data" / "predefined_classes.txt"
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"
CACHE_DB_PATH = ROOT_DIR / "data" / "labelimg-cache.sqlite3"
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".avif",
    ".tif",
    ".tiff",
}
IMAGE_DIRECTORY_NAMES = {"images", "image", "imgs"}
LABEL_DIRECTORY_NAMES = ("labels", "label", "annotations", "annotation")
NO_CACHE_HTML_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}
IMMUTABLE_IMAGE_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
}


@dataclass(slots=True)
class SessionImage:
    id: str
    name: str
    relative_path: str
    full_path: Path
    file_size: int
    mtime_ns: int
    annotation_path: Path | None = None
    annotation_format: str | None = None
    annotation_count: int = 0


@dataclass(slots=True)
class LocalSession:
    id: str
    label: str
    root_path: Path
    images: list[SessionImage]
    images_by_id: dict[str, SessionImage]


@dataclass(slots=True)
class SessionOpenJob:
    id: str
    status: str
    phase: str
    processed_images: int = 0
    total_images: int = 0
    session_revision: int = 0
    session_payload: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class YoloClassSource:
    names: list[str]
    source_path: Path | None = None
    source_mtime_ns: int | None = None


class PathRequest(BaseModel):
    path: str


class RecentDatasetEntry(BaseModel):
    path: str
    label: str


class PersistedSessionStatePayload(BaseModel):
    sourceKind: Literal["image", "dataset"]
    sourcePath: str
    currentImageRelativePath: str | None = None


class AppStatePayload(BaseModel):
    sidebarVisible: bool | None = None
    recentDatasets: list[RecentDatasetEntry] | None = None
    sessionState: PersistedSessionStatePayload | None = None
    hotkeys: dict[str, list[str]] | None = None


LOCAL_SESSIONS: dict[str, LocalSession] = {}
LOCAL_SESSION_JOBS: dict[str, SessionOpenJob] = {}
LOCAL_IMAGE_PATHS: dict[str, Path] = {}
SESSION_JOB_LOCK = Lock()
CACHE_STORE = CacheStore(CACHE_DB_PATH)

app = FastAPI(
    title="labelImg Next",
    version=APP_VERSION,
    description="Backend and static app host for the browser-based labelImg rewrite.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def api_root():
    return {
        "service": "labelimg-next",
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "labelimg-next",
        "version": APP_VERSION,
        "modelReady": False,
    }


@app.get("/api/classes")
def get_predefined_classes():
    if not PREDEFINED_CLASSES_FILE.exists():
        return []

    return [
        line.strip()
        for line in PREDEFINED_CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.get("/api/app-state")
def get_app_state():
    return CACHE_STORE.load_app_state()


@app.put("/api/app-state")
def update_app_state(payload: AppStatePayload):
    patch = payload.model_dump(exclude_unset=True, mode="json")
    return CACHE_STORE.merge_app_state(patch)


@app.post("/api/local/sessions/open-image")
async def open_local_image():
    selected_path = choose_local_image()
    if selected_path is None:
        return {"cancelled": True}

    return build_single_image_session(Path(selected_path))


@app.post("/api/local/sessions/open-directory")
async def open_local_directory():
    selected_path = choose_local_directory()
    if selected_path is None:
        return {"cancelled": True}

    job = start_directory_open_job(Path(selected_path))
    return {
        "cancelled": False,
        "jobId": job.id,
    }


@app.post("/api/local/sessions/open-image-path")
def open_local_image_path(payload: PathRequest):
    path = Path(payload.path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image file")

    return build_single_image_session(path)


@app.post("/api/local/sessions/open-directory-path")
def open_local_directory_path(payload: PathRequest):
    path = Path(payload.path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    return build_directory_session(path)


@app.post("/api/local/sessions/open-directory-path-job")
def open_local_directory_path_job(payload: PathRequest):
    path = Path(payload.path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    job = start_directory_open_job(path)
    return {
        "cancelled": False,
        "jobId": job.id,
    }


@app.get("/api/local/session-jobs/{job_id}")
def read_local_session_job(job_id: str, after_revision: int = 0):
    job = LOCAL_SESSION_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Session job not found")

    session_payload = (
        job.session_payload
        if job.session_payload is not None and job.session_revision > after_revision
        else None
    )

    return {
        "jobId": job.id,
        "status": job.status,
        "phase": job.phase,
        "processedImages": job.processed_images,
        "totalImages": job.total_images,
        "sessionRevision": job.session_revision,
        "session": session_payload,
        "error": job.error,
    }


@app.get("/api/local/images/{image_id}")
def read_local_image(image_id: str):
    image_path = LOCAL_IMAGE_PATHS.get(image_id)
    if image_path is None or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return build_cached_image_response(image_path)


@app.get("/api/local/sessions/{session_id}/images/{image_id}")
def read_local_session_image(session_id: str, image_id: str):
    session = LOCAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    image = session.images_by_id.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return build_cached_image_response(image.full_path)


@app.get("/api/local/sessions/{session_id}/annotations/{image_id}")
def read_local_session_annotations(session_id: str, image_id: str):
    session = LOCAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    image = session.images_by_id.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    annotations = load_image_annotations(session, image)
    image.annotation_count = len(annotations)
    CACHE_STORE.update_dataset_annotation_metadata(
        session.root_path,
        image.relative_path,
        image.annotation_count,
        image.annotation_format,
    )

    return {
        "format": image.annotation_format,
        "count": image.annotation_count,
        "annotations": annotations,
    }


@app.get("/", response_class=HTMLResponse)
def serve_root():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers=NO_CACHE_HTML_HEADERS)

    return HTMLResponse(
        "<h1>Frontend not built. Run npm run build in frontend/ first.</h1>"
    )


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    requested_path = (FRONTEND_DIST_DIR / full_path).resolve()

    try:
        requested_path.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not Found") from exc

    if requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers=NO_CACHE_HTML_HEADERS)

    raise HTTPException(status_code=404, detail="Not Found")


def build_cached_image_response(image_path: Path):
    try:
        stat_result = image_path.stat()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    return FileResponse(
        image_path,
        headers=IMMUTABLE_IMAGE_HEADERS,
        stat_result=stat_result,
    )


def create_local_session(root_path: Path, label: str):
    session = LocalSession(
        id=uuid4().hex,
        label=label,
        root_path=root_path,
        images=[],
        images_by_id={},
    )
    LOCAL_SESSIONS[session.id] = session
    return session


def register_session_image(image: SessionImage):
    LOCAL_IMAGE_PATHS[image.id] = image.full_path


def register_session_images(images: list[SessionImage]):
    for image in images:
        register_session_image(image)


def build_cached_directory_session(root_path: Path):
    manifest = CACHE_STORE.load_dataset_manifest(root_path)
    if manifest is None:
        return None

    session = create_local_session(root_path, manifest["label"])
    session.images = [
        build_cached_session_image(root_path, manifest_image)
        for manifest_image in manifest["images"]
    ]
    session.images_by_id = {image.id: image for image in session.images}
    register_session_images(session.images)
    return session


def build_cached_session_image(root_path: Path, manifest_image: dict[str, Any]):
    annotation_path = manifest_image.get("annotation_path")
    return SessionImage(
        id=manifest_image["id"],
        name=manifest_image["name"],
        relative_path=manifest_image["relative_path"],
        full_path=Path(manifest_image["full_path"]),
        file_size=int(manifest_image["file_size"]),
        mtime_ns=int(manifest_image["mtime_ns"]),
        annotation_path=Path(annotation_path) if annotation_path else None,
        annotation_format=manifest_image.get("annotation_format"),
        annotation_count=int(manifest_image.get("annotation_count") or 0),
    )


def build_single_image_session(path: Path):
    resolved_path = path.expanduser().resolve()
    image = build_session_image(resolved_path, resolved_path.parent)
    session = create_local_session(
        resolved_path.parent,
        resolved_path.stem or "Single image",
    )
    session.images = [image]
    session.images_by_id = {image.id: image}
    register_session_images(session.images)
    return serialize_session(session)


def build_directory_session(
    root_path: Path,
    progress_callback: Callable[..., None] | None = None,
):
    resolved_root = root_path.expanduser().resolve()
    cached_session = build_cached_directory_session(resolved_root)
    if cached_session is not None:
        return serialize_session(cached_session)

    if progress_callback is not None:
        progress_callback(phase="indexing", processed=0, total=0)

    session = create_local_session(
        resolved_root,
        resolved_root.name or "Folder session",
    )
    return scan_directory_session(
        session,
        progress_callback=progress_callback,
        publish_intermediate=progress_callback is not None,
    )


def start_directory_open_job(root_path: Path):
    resolved_root = root_path.expanduser().resolve()
    job = SessionOpenJob(
        id=uuid4().hex,
        status="running",
        phase="indexing",
    )
    LOCAL_SESSION_JOBS[job.id] = job
    cached_session = build_cached_directory_session(resolved_root)

    if cached_session is None:
        session = create_local_session(
            resolved_root,
            resolved_root.name or "Folder session",
        )
    else:
        session = cached_session
        with SESSION_JOB_LOCK:
            job.total_images = len(session.images)
            job.session_payload = serialize_session(session)
            job.session_revision += 1

    def update_progress(
        *,
        phase: str,
        processed: int,
        total: int,
        session: dict[str, Any] | None = None,
    ):
        with SESSION_JOB_LOCK:
            job.phase = phase
            job.processed_images = processed
            job.total_images = total
            if session is not None:
                job.session_payload = session
                job.session_revision += 1

    def worker():
        try:
            session_payload = scan_directory_session(
                session,
                progress_callback=update_progress,
                publish_intermediate=cached_session is None,
            )
            with SESSION_JOB_LOCK:
                job.status = "completed"
                job.phase = "completed"
                job.processed_images = len(session.images)
                job.total_images = len(session.images)
                job.session_payload = session_payload
                job.session_revision += 1
        except Exception as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            with SESSION_JOB_LOCK:
                job.status = "failed"
                job.phase = "failed"
                job.error = detail or "Failed to open directory"

    Thread(target=worker, daemon=True).start()
    return job


def scan_directory_session(
    session: LocalSession,
    *,
    progress_callback: Callable[..., None] | None = None,
    publish_intermediate: bool,
):
    next_images: list[SessionImage] = []
    published_count = 0
    last_publish_at = time.monotonic()

    for path in iter_directory_images(session.root_path):
        next_image = build_session_image(path, session.root_path)
        next_images.append(next_image)
        register_session_image(next_image)

        processed_images = len(next_images)
        should_publish_session = publish_intermediate and (
            processed_images <= 12
            or processed_images - published_count >= 24
            or time.monotonic() - last_publish_at >= 0.2
        )

        if should_publish_session:
            session.images = list(next_images)
            session.images_by_id = {image.id: image for image in next_images}
            published_count = processed_images
            last_publish_at = time.monotonic()

        if progress_callback is not None:
            progress_callback(
                phase="indexing",
                processed=processed_images,
                total=processed_images,
                session=(
                    serialize_session(session) if should_publish_session else None
                ),
            )

    session.images = next_images
    session.images_by_id = {image.id: image for image in next_images}
    CACHE_STORE.save_dataset_manifest(
        session.root_path,
        session.label,
        serialize_manifest_images(next_images),
    )
    return serialize_session(session)


def serialize_session(session: LocalSession):
    return {
        "cancelled": False,
        "sessionId": session.id,
        "sessionLabel": session.label,
        "rootPath": str(session.root_path),
        "images": [
            {
                "id": image.id,
                "name": image.name,
                "relativePath": image.relative_path,
                "annotationCount": image.annotation_count,
                "annotationFormat": image.annotation_format,
            }
            for image in session.images
        ],
    }


def serialize_manifest_images(images: list[SessionImage]):
    return [
        {
            "id": image.id,
            "name": image.name,
            "relative_path": image.relative_path,
            "full_path": str(image.full_path),
            "file_size": image.file_size,
            "mtime_ns": image.mtime_ns,
            "annotation_path": (
                str(image.annotation_path) if image.annotation_path else None
            ),
            "annotation_format": image.annotation_format,
            "annotation_count": image.annotation_count,
        }
        for image in images
    ]


def build_session_image(path: Path, root_path: Path):
    file_stat = path.stat()
    relative_path = path.relative_to(root_path).as_posix()
    annotation_path, annotation_format, annotation_count = find_annotation_sidecar(
        path, root_path
    )
    return SessionImage(
        id=CACHE_STORE.image_id_for_path(
            root_path,
            relative_path,
            mtime_ns=file_stat.st_mtime_ns,
            file_size=file_stat.st_size,
        ),
        name=path.name,
        relative_path=relative_path,
        full_path=path,
        file_size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        annotation_path=annotation_path,
        annotation_format=annotation_format,
        annotation_count=annotation_count,
    )


def find_annotation_sidecar(image_path: Path, session_root: Path):
    for annotation_base in iter_annotation_bases(image_path, session_root):
        xml_path = annotation_base.parent / f"{annotation_base.name}.xml"
        if xml_path.is_file():
            return xml_path, "voc", 1

        txt_path = annotation_base.parent / f"{annotation_base.name}.txt"
        if txt_path.is_file():
            return txt_path, "yolo", 1

    return None, None, 0


def iter_annotation_bases(image_path: Path, session_root: Path):
    seen: set[str] = set()

    def add_candidate(path: Path):
        path_key = str(path)
        if path_key not in seen:
            seen.add(path_key)
            yield path

    image_base = image_path.parent / image_path.stem
    yield from add_candidate(image_base)

    try:
        relative_base = image_path.relative_to(session_root)
    except ValueError:
        relative_base = Path(image_path.name)

    relative_base = relative_base.parent / image_path.stem
    root_name = session_root.name.lower()

    for label_dir in LABEL_DIRECTORY_NAMES:
        yield from add_candidate(session_root / label_dir / relative_base)

        if root_name in IMAGE_DIRECTORY_NAMES:
            yield from add_candidate(session_root.parent / label_dir / relative_base)

    relative_parts = list(relative_base.parts)
    for index, part in enumerate(relative_parts):
        if part.lower() not in IMAGE_DIRECTORY_NAMES:
            continue

        prefix = relative_parts[:index]
        suffix = relative_parts[index + 1 :]
        for label_dir in LABEL_DIRECTORY_NAMES:
            yield from add_candidate(
                session_root / Path(*prefix, label_dir, *suffix)
            )

    absolute_parts = list(image_base.parts)
    for index, part in enumerate(absolute_parts):
        if part.lower() not in IMAGE_DIRECTORY_NAMES:
            continue

        prefix = absolute_parts[:index]
        suffix = absolute_parts[index + 1 :]
        for label_dir in LABEL_DIRECTORY_NAMES:
            yield from add_candidate(Path(*prefix, label_dir, *suffix))

def load_image_annotations(session: LocalSession, image: SessionImage):
    if image.annotation_path is None or image.annotation_format is None:
        return []

    try:
        annotation_mtime_ns = image.annotation_path.stat().st_mtime_ns
    except OSError:
        return []

    if image.annotation_format == "voc":
        cached_annotations = CACHE_STORE.load_annotation_cache(
            annotation_path=image.annotation_path,
            annotation_format="voc",
            annotation_mtime_ns=annotation_mtime_ns,
            image_path=image.full_path,
            image_mtime_ns=image.mtime_ns,
        )
        if cached_annotations is not None:
            return cached_annotations

        annotations = load_pascal_voc_annotations(image.annotation_path)
        CACHE_STORE.save_annotation_cache(
            annotation_path=image.annotation_path,
            annotation_format="voc",
            annotation_mtime_ns=annotation_mtime_ns,
            image_path=image.full_path,
            image_mtime_ns=image.mtime_ns,
            payload=annotations,
        )
        return annotations

    if image.annotation_format == "yolo":
        class_source = load_yolo_classes(image.annotation_path.parent, session.root_path)
        cached_annotations = CACHE_STORE.load_annotation_cache(
            annotation_path=image.annotation_path,
            annotation_format="yolo",
            annotation_mtime_ns=annotation_mtime_ns,
            image_path=image.full_path,
            image_mtime_ns=image.mtime_ns,
            class_source_path=class_source.source_path,
            class_source_mtime_ns=class_source.source_mtime_ns,
        )
        if cached_annotations is not None:
            return cached_annotations

        annotations = load_yolo_annotations(
            annotation_path=image.annotation_path,
            image_path=image.full_path,
            class_source=class_source,
        )
        CACHE_STORE.save_annotation_cache(
            annotation_path=image.annotation_path,
            annotation_format="yolo",
            annotation_mtime_ns=annotation_mtime_ns,
            image_path=image.full_path,
            image_mtime_ns=image.mtime_ns,
            class_source_path=class_source.source_path,
            class_source_mtime_ns=class_source.source_mtime_ns,
            payload=annotations,
        )
        return annotations

    return []


def load_pascal_voc_annotations(annotation_path: Path):
    try:
        root = ElementTree.parse(annotation_path).getroot()
    except Exception:
        return []

    annotations = []
    for object_node in root.findall("object"):
        label = (object_node.findtext("name") or "object").strip() or "object"
        difficult = bool(int(object_node.findtext("difficult") or "0"))
        box = object_node.find("bndbox")
        if box is None:
            continue

        try:
            x_min = float(box.findtext("xmin") or "0")
            y_min = float(box.findtext("ymin") or "0")
            x_max = float(box.findtext("xmax") or "0")
            y_max = float(box.findtext("ymax") or "0")
        except ValueError:
            continue

        annotations.append(
            {
                "id": uuid4().hex,
                "label": label,
                "difficult": difficult,
                "x": x_min,
                "y": y_min,
                "width": max(0.0, x_max - x_min),
                "height": max(0.0, y_max - y_min),
            }
        )

    return annotations


def load_yolo_annotations(
    annotation_path: Path,
    image_path: Path,
    class_source: YoloClassSource,
):
    try:
        image_width, image_height = read_image_size(image_path)
        class_names = class_source.names
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    annotations = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        try:
            class_index = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        x_min = max(x_center - width / 2, 0.0)
        y_min = max(y_center - height / 2, 0.0)
        x_max = min(x_center + width / 2, 1.0)
        y_max = min(y_center + height / 2, 1.0)

        left = round(image_width * x_min)
        top = round(image_height * y_min)
        right = round(image_width * x_max)
        bottom = round(image_height * y_max)
        label = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"class_{class_index}"
        )

        annotations.append(
            {
                "id": uuid4().hex,
                "label": label,
                "difficult": False,
                "x": left,
                "y": top,
                "width": max(0, right - left),
                "height": max(0, bottom - top),
            }
        )

    return annotations


def read_image_size(image_path: Path):
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Pillow is required to read YOLO annotation sizes",
        ) from exc

    with Image.open(image_path) as image:
        return image.size


def load_yolo_classes(labels_dir: Path, session_root: Path):
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path):
        path_key = str(path)
        if path_key not in seen:
            seen.add(path_key)
            candidates.append(path)

    stop_dir = (
        session_root.parent
        if session_root.name.lower() in IMAGE_DIRECTORY_NAMES
        else session_root
    )

    current_dir = labels_dir
    while True:
        add_candidate(current_dir / "classes.txt")
        if current_dir == stop_dir or current_dir.parent == current_dir:
            break
        current_dir = current_dir.parent

    add_candidate(session_root / "classes.txt")
    for label_dir in LABEL_DIRECTORY_NAMES:
        add_candidate(session_root / label_dir / "classes.txt")

    if session_root.name.lower() in IMAGE_DIRECTORY_NAMES:
        add_candidate(session_root.parent / "classes.txt")
        for label_dir in LABEL_DIRECTORY_NAMES:
            add_candidate(session_root.parent / label_dir / "classes.txt")

    add_candidate(PREDEFINED_CLASSES_FILE)

    for candidate in candidates:
        if not candidate.is_file():
            continue

        try:
            names = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return YoloClassSource(
                names=names,
                source_path=candidate,
                source_mtime_ns=candidate.stat().st_mtime_ns,
            )
        except Exception:
            continue

    return YoloClassSource(names=[])


def collect_directory_images(root_path: Path):
    return list(iter_directory_images(root_path))


def iter_directory_images(root_path: Path):
    for current_root, dir_names, file_names in os.walk(root_path):
        dir_names.sort(key=natural_sort_key)
        file_names.sort(key=natural_sort_key)

        current_root_path = Path(current_root)
        for file_name in file_names:
            path = current_root_path / file_name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def natural_sort_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def choose_local_directory():
    dialog = load_tkinter_dialogs()
    root = dialog["root_factory"]()
    try:
        return dialog["filedialog"].askdirectory(parent=root, mustexist=True) or None
    finally:
        root.destroy()


def choose_local_image():
    dialog = load_tkinter_dialogs()
    root = dialog["root_factory"]()
    filetypes = [
        (
            "Images",
            " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS)),
        )
    ]
    try:
        return (
            dialog["filedialog"].askopenfilename(parent=root, filetypes=filetypes)
            or None
        )
    finally:
        root.destroy()


def load_tkinter_dialogs() -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Tkinter file dialogs are not available in this Python runtime",
        ) from exc

    def create_hidden_root():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        return root

    return {
        "filedialog": filedialog,
        "root_factory": create_hidden_root,
    }
