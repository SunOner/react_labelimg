import base64
import hashlib
import html
import importlib.util
import json
import logging
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from uuid import uuid4
from xml.etree import ElementTree

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .cache_store import CacheStore

APP_VERSION = "0.1.0"
LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
PREDEFINED_CLASSES_FILE = ROOT_DIR / "data" / "predefined_classes.txt"
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"
CACHE_DB_PATH = ROOT_DIR / "data" / "labelimg-cache.sqlite3"
PLUGIN_MODELS_DIR = ROOT_DIR / "models" / "plugins"
PLUGIN_RUNTIME_SOURCES_DIR = ROOT_DIR / "data" / "plugin-runtime-sources"
HF_OAUTH_DEFAULT_SCOPES = "openid profile gated-repos"
HF_OAUTH_PROVIDER_URL = "https://huggingface.co"
HF_OAUTH_AUTHORIZE_URL = f"{HF_OAUTH_PROVIDER_URL}/oauth/authorize"
HF_OAUTH_TOKEN_URL = f"{HF_OAUTH_PROVIDER_URL}/oauth/token"
HF_WHOAMI_URL = f"{HF_OAUTH_PROVIDER_URL}/api/whoami-v2"
HF_OAUTH_CALLBACK_PATH = "/api/hf-auth/callback"
HF_OAUTH_SECRET_KEY = "huggingface_oauth"
HF_OAUTH_CONFIG_KEY = "huggingface_oauth_config"
PLUGIN_RUNTIME_STATE_KEY_PREFIX = "plugin_runtime_state:"
HF_OAUTH_PENDING_TTL_SECONDS = 600
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
SAM3_SOURCE_ARCHIVE_URL = (
    "https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip"
)
SAM3_RUNTIME_EXTRA_DEPENDENCIES = (
    "einops",
)
PIP_IPV4_WRAPPER_CODE = """
import runpy
import socket
import sys

_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_GETHOSTBYNAME = socket.gethostbyname


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and family in (0, socket.AF_UNSPEC, socket.AF_INET):
        try:
            ipv4_address = _ORIGINAL_GETHOSTBYNAME(host)
        except OSError:
            return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

        socket_type = type or socket.SOCK_STREAM
        protocol = proto or socket.IPPROTO_TCP
        return [(socket.AF_INET, socket_type, protocol, "", (ipv4_address, port))]

    return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo
sys.argv = ["pip", *sys.argv[1:]]
runpy.run_module("pip", run_name="__main__")
""".strip()
WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"^(?P<drive>[a-zA-Z]):[\\/](?P<rest>.*)$")
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


@dataclass(slots=True)
class PluginDownloadJob:
    plugin_id: str
    status: Literal["running", "completed", "failed"]
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(slots=True)
class PluginRuntimeInstallJob:
    plugin_id: str
    status: Literal["running", "completed", "failed"]
    requested_profile: Literal["auto", "cuda", "cpu"] = "auto"
    resolved_profile: Literal["cuda", "cpu"] | None = None
    step: str | None = None
    step_started_at: str | None = None
    log: str = ""
    message: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class MissingSamRuntimeDependencyError(RuntimeError):
    def __init__(self, dependency: str):
        self.dependency = dependency
        super().__init__(
            "SAM 3 runtime is installed but missing dependency "
            f"'{dependency}'. Re-run the runtime installer in Manage Plugins.",
        )


@dataclass(frozen=True, slots=True)
class PluginModelAsset:
    filename: str
    provider: Literal["huggingface"]
    download_url: str
    repo_id: str
    access_url: str
    docs_url: str
    requires_auth: bool = False
    expected_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class PluginDefinition:
    id: str
    name: str
    version: str
    summary: str
    description: str
    capabilities: tuple[str, ...]
    integration_target: str
    model: PluginModelAsset


class PathRequest(BaseModel):
    path: str


class RecentDatasetEntry(BaseModel):
    path: str
    label: str


class PersistedSessionStatePayload(BaseModel):
    sourceKind: Literal["image", "dataset"]
    sourcePath: str
    currentImageRelativePath: str | None = None


class SamPromptPairPayload(BaseModel):
    prompt: str | None = None
    label: str | None = None


class SamSettingsPayload(BaseModel):
    entries: list[SamPromptPairPayload] | None = None
    scoreThreshold: str | None = None
    maxResults: str | None = None


class AppStatePayload(BaseModel):
    sidebarVisible: bool | None = None
    recentDatasets: list[RecentDatasetEntry] | None = None
    sessionState: PersistedSessionStatePayload | None = None
    hotkeys: dict[str, list[str]] | None = None
    projectClassesByRootPath: dict[str, list[str]] | None = None
    samSettings: SamSettingsPayload | None = None


class LocalAnnotationPayload(BaseModel):
    label: str | None = None
    difficult: bool = False
    x: float
    y: float
    width: float
    height: float


class SaveLocalAnnotationsPayload(BaseModel):
    annotations: list[LocalAnnotationPayload]
    projectClasses: list[str] | None = None


class PluginDownloadRequest(BaseModel):
    force: bool = False


class PluginRuntimeInstallRequest(BaseModel):
    profile: Literal["auto", "cuda", "cpu"] = "auto"


class HuggingFaceAuthConfigPayload(BaseModel):
    clientId: str | None = None


class PluginAnnotationRectPayload(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PluginAutoAnnotateRequest(BaseModel):
    sessionId: str | None = None
    imageId: str
    prompt: str
    label: str | None = None
    mode: Literal["full-image", "selected-box"] = "full-image"
    region: PluginAnnotationRectPayload | None = None
    scoreThreshold: float = 0.2
    maxResults: int = 12


LOCAL_SESSIONS: dict[str, LocalSession] = {}
LOCAL_SESSION_JOBS: dict[str, SessionOpenJob] = {}
LOCAL_IMAGE_PATHS: dict[str, Path] = {}
SESSION_JOB_LOCK = Lock()
HF_OAUTH_PENDING_STATES: dict[str, dict[str, Any]] = {}
HF_OAUTH_PENDING_LOCK = Lock()
PLUGIN_DOWNLOAD_JOBS: dict[str, PluginDownloadJob] = {}
PLUGIN_DOWNLOAD_LOCK = Lock()
PLUGIN_RUNTIME_INSTALL_JOBS: dict[str, PluginRuntimeInstallJob] = {}
PLUGIN_RUNTIME_INSTALL_LOCK = Lock()
PLUGIN_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
PLUGIN_RUNTIME_LOCK = Lock()
CACHE_STORE = CacheStore(CACHE_DB_PATH)
PLUGIN_DEFINITIONS = {
    "sam-3-1": PluginDefinition(
        id="sam-3-1",
        name="SAM 3.1",
        version="2026-03-27",
        summary="Prompt-driven segmentation model for refining boxes into masks.",
        description=(
            "Meta's SAM 3.1 can segment images from text, box, point and mask "
            "prompts. In this app it is prepared as the first plugin for "
            "prompt-guided box segmentation workflows."
        ),
        capabilities=(
            "Text prompts",
            "Box prompts",
            "Mask generation",
            "Prompt-guided refinement",
        ),
        integration_target="Segmentation of boxes by prompts",
        model=PluginModelAsset(
            filename="sam3.1_multiplex.pt",
            provider="huggingface",
            download_url=(
                "https://huggingface.co/facebook/sam3.1/resolve/main/"
                "sam3.1_multiplex.pt?download=true"
            ),
            repo_id="facebook/sam3.1",
            access_url="https://huggingface.co/facebook/sam3.1",
            docs_url="https://github.com/facebookresearch/sam3",
            requires_auth=True,
            expected_size_bytes=3_500_000_000,
        ),
    ),
}

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


@app.get("/api/hf-auth/status")
def get_hf_auth_status(request: Request):
    return serialize_hf_auth_state(request)


@app.put("/api/hf-auth/config")
def update_hf_auth_config(request: Request, payload: HuggingFaceAuthConfigPayload):
    next_client_id = (payload.clientId or "").strip()
    if next_client_id:
        CACHE_STORE.save_service_secret(
            HF_OAUTH_CONFIG_KEY,
            {
                "clientId": next_client_id,
                "updatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
    else:
        CACHE_STORE.delete_service_secret(HF_OAUTH_CONFIG_KEY)
        CACHE_STORE.delete_service_secret(HF_OAUTH_SECRET_KEY)

    return serialize_hf_auth_state(request)


@app.post("/api/hf-auth/start")
def start_hf_auth(request: Request):
    client_id = get_hf_oauth_client_id()
    redirect_uri = build_hf_oauth_redirect_uri(request)
    state = secrets.token_urlsafe(24)
    code_verifier = generate_pkce_code_verifier()
    code_challenge = build_pkce_code_challenge(code_verifier)

    with HF_OAUTH_PENDING_LOCK:
        prune_expired_hf_oauth_states()
        HF_OAUTH_PENDING_STATES[state] = {
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "created_at": time.time(),
        }

    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": get_hf_oauth_scopes(),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return {
        "authorizationUrl": f"{HF_OAUTH_AUTHORIZE_URL}?{query}",
        "redirectUri": redirect_uri,
        "scopes": get_hf_oauth_scopes().split(),
        "clientId": client_id,
    }


@app.get(HF_OAUTH_CALLBACK_PATH, response_class=HTMLResponse)
def hf_auth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    if error:
        return build_hf_oauth_callback_page(
            success=False,
            message=error_description or error,
        )

    if not code or not state:
        return build_hf_oauth_callback_page(
            success=False,
            message="Missing authorization code.",
        )

    with HF_OAUTH_PENDING_LOCK:
        prune_expired_hf_oauth_states()
        pending = HF_OAUTH_PENDING_STATES.pop(state, None)

    if pending is None:
        return build_hf_oauth_callback_page(
            success=False,
            message="The login session expired. Please try again.",
        )

    try:
        token_payload = exchange_hf_oauth_code(
            code=code,
            code_verifier=str(pending["code_verifier"]),
            redirect_uri=str(pending["redirect_uri"]),
        )
        store_hf_oauth_token(token_payload)
    except HTTPException as exc:
        return build_hf_oauth_callback_page(success=False, message=exc.detail)

    return build_hf_oauth_callback_page(
        success=True,
        message="Hugging Face account connected.",
    )


@app.post("/api/hf-auth/logout")
def logout_hf_auth(request: Request):
    CACHE_STORE.delete_service_secret(HF_OAUTH_SECRET_KEY)
    return serialize_hf_auth_state(request)


@app.get("/api/plugins")
def list_plugins():
    return [serialize_plugin(plugin) for plugin in PLUGIN_DEFINITIONS.values()]


@app.post("/api/plugins/{plugin_id}/download-model")
def download_plugin_model(plugin_id: str, payload: PluginDownloadRequest | None = None):
    plugin = get_plugin_definition(plugin_id)
    model_path = get_plugin_model_path(plugin)
    request_payload = payload or PluginDownloadRequest()

    if model_path.is_file() and not request_payload.force:
        return serialize_plugin(plugin)

    auth_token = resolve_huggingface_access_token()
    if plugin.model.requires_auth and not auth_token:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{plugin.name} requires approved Hugging Face access. "
                "Connect a Hugging Face account in the Plugins panel first."
            ),
        )

    with PLUGIN_DOWNLOAD_LOCK:
        current_job = PLUGIN_DOWNLOAD_JOBS.get(plugin.id)
        if current_job is not None and current_job.status == "running":
            return serialize_plugin(plugin)

        PLUGIN_DOWNLOAD_JOBS[plugin.id] = PluginDownloadJob(
            plugin_id=plugin.id,
            status="running",
            total_bytes=plugin.model.expected_size_bytes,
            started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    def worker():
        run_plugin_download_job(
            plugin,
            destination=model_path,
            auth_token=auth_token,
        )

    Thread(target=worker, daemon=True).start()
    return serialize_plugin(plugin)


@app.post("/api/plugins/{plugin_id}/install-runtime")
def install_plugin_runtime(
    plugin_id: str,
    payload: PluginRuntimeInstallRequest | None = None,
):
    plugin = get_plugin_definition(plugin_id)
    request_payload = payload or PluginRuntimeInstallRequest()

    if plugin.id != "sam-3-1":
        raise HTTPException(
            status_code=400,
            detail=f"{plugin.name} does not expose a runtime installer in this app yet.",
        )

    with PLUGIN_RUNTIME_INSTALL_LOCK:
        current_job = PLUGIN_RUNTIME_INSTALL_JOBS.get(plugin.id)
        if current_job is not None and current_job.status == "running":
            return serialize_plugin(plugin)

        active_job = next(
            (
                job
                for job in PLUGIN_RUNTIME_INSTALL_JOBS.values()
                if job.status == "running" and job.plugin_id != plugin.id
            ),
            None,
        )
        if active_job is not None:
            active_plugin = PLUGIN_DEFINITIONS.get(active_job.plugin_id)
            active_name = active_plugin.name if active_plugin else active_job.plugin_id
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{active_name} runtime installation is already running. "
                    "Wait for it to finish before starting another runtime install."
                ),
            )

        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started_log_timestamp = format_plugin_runtime_log_timestamp()
        PLUGIN_RUNTIME_INSTALL_JOBS[plugin.id] = PluginRuntimeInstallJob(
            plugin_id=plugin.id,
            status="running",
            requested_profile=request_payload.profile,
            step="preparing",
            step_started_at=started_at,
            log=(
                f"[{started_log_timestamp}] Starting SAM runtime installation for {plugin.name}.\n"
                f"[{started_log_timestamp}] Requested profile: {request_payload.profile.upper()}.\n"
            ),
            message="Preparing SAM runtime installation.",
            started_at=started_at,
        )

    def worker():
        run_plugin_runtime_install_job(plugin, request_payload.profile)

    Thread(target=worker, daemon=True).start()
    return serialize_plugin(plugin)


@app.post("/api/plugins/{plugin_id}/auto-annotate")
def auto_annotate_with_plugin(plugin_id: str, payload: PluginAutoAnnotateRequest):
    plugin = get_plugin_definition(plugin_id)

    if plugin.id != "sam-3-1":
        raise HTTPException(
            status_code=400,
            detail=f"{plugin.name} does not support auto-annotation in this app yet.",
        )

    return run_sam_auto_annotation(plugin, payload)


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
    path = resolve_local_request_path(payload.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image file")

    return build_single_image_session(path)


@app.post("/api/local/sessions/open-directory-path")
def open_local_directory_path(payload: PathRequest):
    path = resolve_local_request_path(payload.path)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    return build_directory_session(path)


@app.post("/api/local/sessions/open-directory-path-job")
def open_local_directory_path_job(payload: PathRequest):
    path = resolve_local_request_path(payload.path)
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


@app.put("/api/local/sessions/{session_id}/annotations/{image_id}")
def save_local_session_annotations(
    session_id: str,
    image_id: str,
    payload: SaveLocalAnnotationsPayload,
):
    session = LOCAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    image = session.images_by_id.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    annotations = normalize_annotation_payloads(payload.annotations)
    save_image_annotations(
        session,
        image,
        annotations,
        preferred_classes=payload.projectClasses or [],
    )

    return {
        "format": image.annotation_format,
        "count": image.annotation_count,
        "savedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


@app.delete("/api/local/sessions/{session_id}/images/{image_id}")
def delete_local_session_image(session_id: str, image_id: str):
    session = LOCAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    image = session.images_by_id.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    delete_session_image_files(image)
    remove_image_from_matching_sessions(session.root_path, image.id)
    CACHE_STORE.save_dataset_manifest(
        session.root_path,
        session.label,
        serialize_manifest_images(session.images),
    )
    return serialize_session(session)


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


def load_hf_oauth_config():
    payload = CACHE_STORE.load_service_secret(HF_OAUTH_CONFIG_KEY)
    return payload if isinstance(payload, dict) else None


def get_hf_oauth_client_id(required: bool = True):
    config = load_hf_oauth_config()
    client_id = str(config.get("clientId") or "").strip() if config else ""
    if required and not client_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Hugging Face OAuth is not configured. Add the OAuth Client ID "
                "in the Plugins panel first."
            ),
        )
    return client_id or None


def get_hf_oauth_scopes():
    scopes = HF_OAUTH_DEFAULT_SCOPES.split()
    for required_scope in ("openid", "profile", "gated-repos"):
        if required_scope not in scopes:
            scopes.append(required_scope)
    return " ".join(scopes)


def build_hf_oauth_redirect_uri(request: Request):
    return f"{str(request.base_url).rstrip('/')}{HF_OAUTH_CALLBACK_PATH}"


def resolve_local_request_path(raw_path: str):
    normalized_path = (raw_path or "").strip()
    if not normalized_path:
        return Path(".").resolve()

    candidate_paths = [normalized_path]
    translated_wsl_path = translate_windows_path_for_wsl(normalized_path)
    if translated_wsl_path and translated_wsl_path not in candidate_paths:
        candidate_paths.insert(0, translated_wsl_path)

    for candidate in candidate_paths:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            continue

        if resolved.exists():
            return resolved

    return Path(candidate_paths[0]).expanduser().resolve()


def translate_windows_path_for_wsl(raw_path: str):
    if os.name == "nt":
        return None

    match = WINDOWS_DRIVE_PATH_PATTERN.match(raw_path)
    if not match:
        return None

    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/").lstrip("/")
    if not rest:
        return f"/mnt/{drive}"
    return f"/mnt/{drive}/{rest}"


def generate_pkce_code_verifier():
    return base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode(
        "ascii"
    )


def build_pkce_code_challenge(code_verifier: str):
    return (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def prune_expired_hf_oauth_states():
    now = time.time()
    expired_states = [
        state
        for state, payload in HF_OAUTH_PENDING_STATES.items()
        if now - float(payload.get("created_at", 0)) > HF_OAUTH_PENDING_TTL_SECONDS
    ]
    for state in expired_states:
        HF_OAUTH_PENDING_STATES.pop(state, None)


def exchange_hf_oauth_code(*, code: str, code_verifier: str, redirect_uri: str):
    token_payload = request_json_url(
        HF_OAUTH_TOKEN_URL,
        form_data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": get_hf_oauth_client_id(),
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
    )

    access_token = str(token_payload.get("access_token") or "").strip()
    if not access_token:
        raise HTTPException(
            status_code=502,
            detail="Hugging Face OAuth did not return an access token.",
        )

    user_payload = request_json_url(
        HF_WHOAMI_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    token_payload["user"] = normalize_hf_user_payload(user_payload)
    return token_payload


def normalize_hf_user_payload(payload: Any):
    if not isinstance(payload, dict):
        return None

    organizations = payload.get("orgs") or payload.get("organizations") or []
    org_names = []
    if isinstance(organizations, list):
        for organization in organizations:
            if isinstance(organization, dict):
                name = organization.get("name") or organization.get("preferred_username")
                if isinstance(name, str) and name.strip():
                    org_names.append(name.strip())

    def pick_string(*values: Any):
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    return {
        "username": pick_string(
            payload.get("name"),
            payload.get("preferred_username"),
            payload.get("user"),
        ),
        "fullName": pick_string(
            payload.get("fullname"),
            payload.get("fullName"),
            payload.get("displayName"),
        ),
        "email": pick_string(payload.get("email")),
        "avatarUrl": pick_string(payload.get("avatarUrl"), payload.get("avatar")),
        "organizations": org_names,
    }


def request_json_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
):
    request_headers = {
        "User-Agent": "labelimg-next/0.1.0",
        "Accept": "application/json",
        **(headers or {}),
    }
    request_data: bytes | None = None
    if form_data is not None:
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_data = urlencode(form_data).encode("utf-8")

    request = UrlRequest(url, data=request_data, headers=request_headers)
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = read_http_error_detail(exc)
        raise HTTPException(
            status_code=502 if exc.code >= 500 else exc.code,
            detail=detail or f"Hugging Face request failed with HTTP {exc.code}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach Hugging Face: {exc.reason}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Hugging Face returned an invalid JSON response.",
        ) from exc


def read_http_error_detail(error: HTTPError):
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except Exception:
        return ""

    if isinstance(payload, dict):
        detail = payload.get("error_description") or payload.get("error")
        if isinstance(detail, str):
            return detail
    return ""


def store_hf_oauth_token(token_payload: dict[str, Any]):
    expires_in = token_payload.get("expires_in")
    expires_at: str | None = None
    if isinstance(expires_in, int) and expires_in > 0:
        expires_at = (
            datetime.now().astimezone().timestamp() + expires_in
        )
        expires_at = datetime.fromtimestamp(expires_at).astimezone().isoformat(
            timespec="seconds"
        )

    CACHE_STORE.save_service_secret(
        HF_OAUTH_SECRET_KEY,
        {
            "accessToken": token_payload.get("access_token"),
            "tokenType": token_payload.get("token_type") or "bearer",
            "scope": token_payload.get("scope") or get_hf_oauth_scopes(),
            "expiresAt": expires_at,
            "savedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "user": token_payload.get("user"),
        },
    )


def load_hf_oauth_secret():
    payload = CACHE_STORE.load_service_secret(HF_OAUTH_SECRET_KEY)
    return payload if isinstance(payload, dict) else None


def parse_iso_datetime(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def is_hf_oauth_secret_expired(secret_payload: dict[str, Any] | None):
    if not secret_payload:
        return False
    expires_at = parse_iso_datetime(secret_payload.get("expiresAt"))
    if expires_at is None:
        return False
    return expires_at <= datetime.now(expires_at.tzinfo)


def serialize_hf_auth_state(request: Request):
    secret_payload = load_hf_oauth_secret()
    config_payload = load_hf_oauth_config()
    client_id = str(config_payload.get("clientId") or "").strip() if config_payload else ""
    is_authenticated = bool(
        secret_payload and not is_hf_oauth_secret_expired(secret_payload)
    )
    auth_source = "oauth" if is_authenticated else None

    return {
        "provider": "huggingface",
        "isConfigured": bool(client_id),
        "clientId": client_id or None,
        "callbackUrl": build_hf_oauth_redirect_uri(request),
        "scopes": get_hf_oauth_scopes().split(),
        "isAuthenticated": is_authenticated,
        "isExpired": bool(
            secret_payload and is_hf_oauth_secret_expired(secret_payload)
        ),
        "hasUsableAccessToken": auth_source is not None,
        "authSource": auth_source,
        "user": secret_payload.get("user") if is_authenticated and secret_payload else None,
        "expiresAt": secret_payload.get("expiresAt") if secret_payload else None,
        "savedAt": secret_payload.get("savedAt") if secret_payload else None,
    }


def resolve_huggingface_access_token():
    secret_payload = load_hf_oauth_secret()
    if secret_payload and not is_hf_oauth_secret_expired(secret_payload):
        access_token = secret_payload.get("accessToken")
        if isinstance(access_token, str) and access_token.strip():
            return access_token.strip()
    return None


def build_hf_oauth_callback_page(*, success: bool, message: str):
    escaped_message = html.escape(message)
    payload = json.dumps(
        {
            "type": "hf-oauth-complete",
            "success": success,
            "message": message,
        }
    )
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Hugging Face Login</title>
    <style>
      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #0b1014;
        color: #e8f1ec;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }}
      main {{
        width: min(460px, calc(100vw - 40px));
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
      }}
      h1 {{ margin: 0 0 12px; font-size: 1rem; }}
      p {{ margin: 0; line-height: 1.5; color: rgba(232,241,236,0.8); }}
    </style>
  </head>
  <body>
    <main>
      <h1>{"Connected" if success else "Authorization failed"}</h1>
      <p>{escaped_message}</p>
    </main>
    <script>
      const payload = {payload};
      if (window.opener) {{
        window.opener.postMessage(payload, window.location.origin);
      }}
      setTimeout(() => window.close(), 400);
    </script>
  </body>
</html>
        """.strip(),
        headers=NO_CACHE_HTML_HEADERS,
    )


def get_plugin_definition(plugin_id: str):
    plugin = PLUGIN_DEFINITIONS.get(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return plugin


def get_plugin_model_dir(plugin: PluginDefinition):
    return (PLUGIN_MODELS_DIR / plugin.id).resolve()


def get_plugin_model_path(plugin: PluginDefinition):
    return get_plugin_model_dir(plugin) / plugin.model.filename


def get_plugin_download_job(plugin_id: str):
    with PLUGIN_DOWNLOAD_LOCK:
        return PLUGIN_DOWNLOAD_JOBS.get(plugin_id)


def update_plugin_download_job(plugin_id: str, **changes: Any):
    with PLUGIN_DOWNLOAD_LOCK:
        job = PLUGIN_DOWNLOAD_JOBS.get(plugin_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)


def serialize_plugin_download(plugin: PluginDefinition):
    job = get_plugin_download_job(plugin.id)
    if job is None:
        return {
            "status": "idle",
            "downloadedBytes": 0,
            "totalBytes": plugin.model.expected_size_bytes,
            "error": None,
            "startedAt": None,
            "finishedAt": None,
        }

    return {
        "status": job.status,
        "downloadedBytes": job.downloaded_bytes,
        "totalBytes": job.total_bytes,
        "error": job.error,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
    }


def serialize_plugin_runtime_install(plugin: PluginDefinition):
    job = PLUGIN_RUNTIME_INSTALL_JOBS.get(plugin.id)
    if job is None:
        return {
            "status": "idle",
            "requestedProfile": None,
            "resolvedProfile": None,
            "step": None,
            "stepStartedAt": None,
            "log": None,
            "message": None,
            "error": None,
            "startedAt": None,
            "finishedAt": None,
        }

    return {
        "status": job.status,
        "requestedProfile": job.requested_profile,
        "resolvedProfile": job.resolved_profile,
        "step": job.step,
        "stepStartedAt": job.step_started_at,
        "log": job.log,
        "message": job.message,
        "error": job.error,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
    }


def get_plugin_runtime_state_key(plugin_id: str):
    return f"{PLUGIN_RUNTIME_STATE_KEY_PREFIX}{plugin_id}"


def refresh_python_site_packages():
    candidate_paths: list[str] = []

    try:
        candidate_paths.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    if isinstance(user_site, str) and user_site:
        candidate_paths.append(user_site)

    seen_paths: set[str] = set()
    for path in candidate_paths:
        if not path or path in seen_paths or not os.path.isdir(path):
            continue
        seen_paths.add(path)
        try:
            site.addsitedir(path)
        except Exception:
            continue

    importlib.invalidate_caches()


def load_cached_plugin_runtime_state(plugin: PluginDefinition):
    payload = CACHE_STORE.load_service_secret(get_plugin_runtime_state_key(plugin.id))
    if not isinstance(payload, dict):
        return None

    status = payload.get("status")
    if status not in {"ready", "missing-runtime", "error"}:
        return None

    device = payload.get("device")
    if device not in {"cuda", "cpu", None}:
        device = None

    message = payload.get("message")
    if not isinstance(message, str):
        message = None

    return {
        "status": status,
        "device": device,
        "message": message,
    }


def store_plugin_runtime_state(
    plugin_id: str,
    *,
    status: Literal["ready", "missing-runtime", "error"],
    device: Literal["cuda", "cpu"] | None = None,
    message: str | None = None,
):
    CACHE_STORE.save_service_secret(
        get_plugin_runtime_state_key(plugin_id),
        {
            "status": status,
            "device": device,
            "message": message,
            "checkedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )


def has_sam_runtime_packages():
    refresh_python_site_packages()
    return all(
        importlib.util.find_spec(module_name) is not None
        for module_name in ("torch", "sam3")
    )


def serialize_plugin(plugin: PluginDefinition):
    model_path = get_plugin_model_path(plugin)
    installed_bytes: int | None = None
    installed_at: str | None = None

    if model_path.is_file():
        stat_result = model_path.stat()
        installed_bytes = stat_result.st_size
        installed_at = (
            datetime.fromtimestamp(stat_result.st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        )

    return {
        "id": plugin.id,
        "name": plugin.name,
        "version": plugin.version,
        "summary": plugin.summary,
        "description": plugin.description,
        "capabilities": list(plugin.capabilities),
        "integrationTarget": plugin.integration_target,
        "download": serialize_plugin_download(plugin),
        "runtime": serialize_plugin_runtime(plugin),
        "model": {
            "filename": plugin.model.filename,
            "provider": plugin.model.provider,
            "repoId": plugin.model.repo_id,
            "requiresAuth": plugin.model.requires_auth,
            "expectedSizeBytes": plugin.model.expected_size_bytes,
            "accessUrl": plugin.model.access_url,
            "docsUrl": plugin.model.docs_url,
            "isInstalled": model_path.is_file(),
            "installedBytes": installed_bytes,
            "installedAt": installed_at,
            "path": str(model_path) if model_path.is_file() else None,
        },
    }


def run_plugin_download_job(
    plugin: PluginDefinition,
    *,
    destination: Path,
    auth_token: str | None,
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")
    cleanup_partial_download(partial_path)
    headers = {
        "User-Agent": "labelimg-next/0.1.0",
        "Accept": "application/octet-stream",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    request = UrlRequest(plugin.model.download_url, headers=headers)

    try:
        with urlopen(request, timeout=60) as response, partial_path.open("wb") as file_obj:
            total_bytes = parse_download_total_bytes(response, plugin.model.expected_size_bytes)
            downloaded_bytes = 0
            update_plugin_download_job(
                plugin.id,
                total_bytes=total_bytes,
                downloaded_bytes=0,
                error=None,
            )

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                downloaded_bytes += len(chunk)
                update_plugin_download_job(
                    plugin.id,
                    downloaded_bytes=downloaded_bytes,
                )

        os.replace(partial_path, destination)
        update_plugin_download_job(
            plugin.id,
            status="completed",
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes or downloaded_bytes,
            error=None,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
    except HTTPError as exc:
        cleanup_partial_download(partial_path)
        if exc.code in {401, 403}:
            mark_plugin_download_failed(
                plugin.id,
                (
                    f"Access to {plugin.name} weights was denied by Hugging Face. "
                    "Make sure the connected account has been approved for "
                    f"{plugin.model.repo_id}."
                ),
            )
            return
        mark_plugin_download_failed(
            plugin.id,
            f"Failed to download {plugin.name}: upstream returned HTTP {exc.code}",
        )
        return
    except URLError as exc:
        cleanup_partial_download(partial_path)
        mark_plugin_download_failed(
            plugin.id,
            f"Failed to download {plugin.name}: {exc.reason}",
        )
        return
    except OSError as exc:
        cleanup_partial_download(partial_path)
        mark_plugin_download_failed(
            plugin.id,
            f"Failed to save {plugin.name} model to disk",
        )
        return


def parse_download_total_bytes(response: Any, fallback: int | None = None):
    header_value = None
    try:
        header_value = response.headers.get("Content-Length")
    except Exception:
        header_value = None

    try:
        parsed = int(header_value) if header_value else None
    except (TypeError, ValueError):
        parsed = None
    return parsed or fallback


def mark_plugin_download_failed(plugin_id: str, error_message: str):
    update_plugin_download_job(
        plugin_id,
        status="failed",
        error=error_message,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def cleanup_partial_download(path: Path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def format_plugin_runtime_log_timestamp():
    return datetime.now().astimezone().strftime("%H:%M:%S")


def append_plugin_runtime_install_log(plugin_id: str, text: str):
    if not text:
        return

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    with PLUGIN_RUNTIME_INSTALL_LOCK:
        current = PLUGIN_RUNTIME_INSTALL_JOBS.get(plugin_id)
        if current is None:
            return

        current.log = f"{current.log}{normalized_text}"


def log_plugin_runtime_install_event(plugin_id: str, message: str):
    stripped_message = message.strip()
    if not stripped_message:
        return

    append_plugin_runtime_install_log(
        plugin_id,
        f"[{format_plugin_runtime_log_timestamp()}] {stripped_message}\n",
    )


def update_plugin_runtime_install_job(plugin_id: str, **changes):
    with PLUGIN_RUNTIME_INSTALL_LOCK:
        current = PLUGIN_RUNTIME_INSTALL_JOBS.get(plugin_id)
        if current is None:
            return

        next_step = changes.get("step")
        if (
            next_step is not None
            and next_step != current.step
            and "step_started_at" not in changes
        ):
            changes["step_started_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

        for key, value in changes.items():
            if value is not None or hasattr(current, key):
                setattr(current, key, value)


def mark_plugin_runtime_install_failed(plugin_id: str, error_message: str):
    log_plugin_runtime_install_event(plugin_id, f"ERROR: {error_message}")
    store_plugin_runtime_state(
        plugin_id,
        status="missing-runtime",
        device=None,
        message=error_message,
    )
    update_plugin_runtime_install_job(
        plugin_id,
        status="failed",
        error=error_message,
        step="failed",
        message=None,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def ensure_supported_sam_runtime_host():
    if sys.version_info < (3, 12):
        current_version = platform.python_version()
        raise RuntimeError(
            "SAM 3 requires Python 3.12 or newer in the backend environment. "
            f"Current interpreter: Python {current_version}.",
        )


def ensure_supported_sam_runtime_platform():
    current_platform = platform.system()
    if current_platform != "Linux":
        raise RuntimeError(
            "Official SAM 3 runtime is not supported on native "
            f"{current_platform}. Use Linux or WSL2 with an NVIDIA GPU and CUDA 12.6+.",
        )


def ensure_supported_sam_runtime_profile(
    requested_profile: Literal["auto", "cuda", "cpu"],
):
    if requested_profile == "cpu":
        raise RuntimeError(
            "Official SAM 3 runtime currently requires NVIDIA CUDA 12.6+. "
            "CPU-only installation is not supported by the upstream runtime.",
        )


def detect_nvidia_gpu():
    command = shutil.which("nvidia-smi")
    if not command:
        return False

    try:
        completed = subprocess.run(
            [command, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return bool(completed.stdout.strip())


def resolve_runtime_install_profile(
    requested_profile: Literal["auto", "cuda", "cpu"],
):
    if requested_profile == "auto":
        return "cuda" if detect_nvidia_gpu() else "cpu"
    return requested_profile


def format_process_output_tail(output: str | None, *, max_lines: int = 10):
    if not output:
        return ""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""

    return " | ".join(lines[-max_lines:])


def run_pip_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    runtime_plugin_id: str | None = None,
):
    command = [sys.executable, "-u", "-c", PIP_IPV4_WRAPPER_CODE, *arguments]
    command_display = " ".join(
        shlex.quote(part) for part in [sys.executable, "-u", "-m", "pip", *arguments]
    )
    if runtime_plugin_id:
        log_plugin_runtime_install_event(
            runtime_plugin_id,
            "Using IPv4-only pip networking to avoid WSL IPv6 stalls.",
        )
        append_plugin_runtime_install_log(
            runtime_plugin_id,
            f"\n$ {command_display}\n",
        )

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd or ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start pip: {exc.strerror or exc}") from exc

    if process.stdout is None:
        process.kill()
        raise RuntimeError(f"`{command_display}` did not expose stdout for logging.")

    output_parts: list[str] = []
    try:
        for line in process.stdout:
            output_parts.append(line)
            if runtime_plugin_id:
                append_plugin_runtime_install_log(runtime_plugin_id, line)
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        output = "".join(output_parts)
        summary = format_process_output_tail(output)
        prefix = " ".join(arguments[:4]) or "pip"
        if summary:
            raise RuntimeError(f"`{prefix}` failed: {summary}")
        raise RuntimeError(f"`{prefix}` failed.")


def replace_runtime_source_directory(destination: Path, source_dir: Path):
    destination_parent = destination.parent.resolve()
    runtime_root = PLUGIN_RUNTIME_SOURCES_DIR.resolve()
    if destination_parent != runtime_root and runtime_root not in destination_parent.parents:
        raise RuntimeError("Refusing to replace a runtime source outside the app data directory.")

    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    shutil.move(str(source_dir), str(destination))


def download_sam3_runtime_source(plugin: PluginDefinition):
    source_root = PLUGIN_RUNTIME_SOURCES_DIR / plugin.id
    source_root.mkdir(parents=True, exist_ok=True)
    destination = source_root / "sam3"
    archive_headers = {
        "User-Agent": "labelimg-next/0.1.0",
        "Accept": "application/zip",
    }
    log_plugin_runtime_install_event(
        plugin.id,
        "Downloading official SAM 3 source archive from GitHub.",
    )

    try:
        with tempfile.TemporaryDirectory(dir=str(source_root)) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            archive_path = temp_dir / "sam3.zip"
            extract_dir = temp_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            request = UrlRequest(SAM3_SOURCE_ARCHIVE_URL, headers=archive_headers)

            with urlopen(request, timeout=60) as response, archive_path.open("wb") as file_obj:
                shutil.copyfileobj(response, file_obj)

            log_plugin_runtime_install_event(
                plugin.id,
                "Archive downloaded. Extracting files.",
            )
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_dir)

            extracted_root = next(
                (path for path in extract_dir.iterdir() if path.is_dir()),
                None,
            )
            if extracted_root is None:
                raise RuntimeError("GitHub returned an empty SAM 3 source archive.")

            replace_runtime_source_directory(destination, extracted_root)
            log_plugin_runtime_install_event(
                plugin.id,
                f"SAM 3 source prepared at {destination}.",
            )
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download SAM 3 source from GitHub: HTTP {exc.code}.",
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download SAM 3 source from GitHub: {exc.reason}.",
        ) from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Failed to prepare SAM 3 source files: {exc}.",
        ) from exc

    return destination


def install_sam_runtime_dependencies(
    profile: Literal["cuda", "cpu"],
    *,
    plugin_id: str,
):
    index_url = PYTORCH_CUDA_INDEX_URL if profile == "cuda" else PYTORCH_CPU_INDEX_URL
    run_pip_command(
        [
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
        ],
        runtime_plugin_id=plugin_id,
    )
    run_pip_command(
        [
            "install",
            "--upgrade",
            *SAM3_RUNTIME_EXTRA_DEPENDENCIES,
        ],
        runtime_plugin_id=plugin_id,
    )


def verify_sam_runtime_install(plugin: PluginDefinition):
    attempted_dependencies: set[str] = set()

    while True:
        importlib.invalidate_caches()
        with PLUGIN_RUNTIME_LOCK:
            PLUGIN_RUNTIME_CACHE.pop(plugin.id, None)

        try:
            return resolve_sam_runtime_environment()
        except MissingSamRuntimeDependencyError as exc:
            dependency = exc.dependency.strip()
            if not dependency or dependency in attempted_dependencies:
                raise RuntimeError(str(exc)) from exc

            attempted_dependencies.add(dependency)
            update_plugin_runtime_install_job(
                plugin.id,
                step="installing-runtime-dependency",
                message=f"Installing missing SAM dependency: {dependency}.",
            )
            log_plugin_runtime_install_event(
                plugin.id,
                f"Installing missing SAM dependency: {dependency}.",
            )
            run_pip_command(
                ["install", "--upgrade", dependency],
                runtime_plugin_id=plugin.id,
            )


def run_plugin_runtime_install_job(
    plugin: PluginDefinition,
    requested_profile: Literal["auto", "cuda", "cpu"],
):
    try:
        log_plugin_runtime_install_event(plugin.id, "Checking runtime prerequisites.")
        ensure_supported_sam_runtime_host()
        ensure_supported_sam_runtime_platform()
        ensure_supported_sam_runtime_profile(requested_profile)
        resolved_profile = resolve_runtime_install_profile(requested_profile)
        log_plugin_runtime_install_event(
            plugin.id,
            f"Resolved installation profile: {resolved_profile.upper()}.",
        )
        update_plugin_runtime_install_job(
            plugin.id,
            resolved_profile=resolved_profile,
            step="installing-pytorch",
            message=(
                "Installing PyTorch with CUDA support."
                if resolved_profile == "cuda"
                else "Installing CPU-only PyTorch runtime."
            ),
            error=None,
        )
        log_plugin_runtime_install_event(
            plugin.id,
            "Installing PyTorch runtime packages.",
        )
        install_sam_runtime_dependencies(
            resolved_profile,
            plugin_id=plugin.id,
        )

        update_plugin_runtime_install_job(
            plugin.id,
            step="downloading-sam3",
            message="Downloading official SAM 3 Python package source.",
        )
        source_dir = download_sam3_runtime_source(plugin)

        update_plugin_runtime_install_job(
            plugin.id,
            step="installing-sam3",
            message="Installing SAM 3 Python package.",
        )
        log_plugin_runtime_install_event(
            plugin.id,
            "Installing SAM 3 package in editable mode.",
        )
        run_pip_command(
            ["install", "--upgrade", "-e", str(source_dir)],
            cwd=ROOT_DIR,
            runtime_plugin_id=plugin.id,
        )

        update_plugin_runtime_install_job(
            plugin.id,
            step="verifying",
            message="Verifying SAM runtime imports.",
        )
        log_plugin_runtime_install_event(
            plugin.id,
            "Verifying that the runtime imports successfully.",
        )
        runtime = verify_sam_runtime_install(plugin)
        device = "cuda" if runtime["torch"].cuda.is_available() else "cpu"
        if resolved_profile == "cuda" and device != "cuda":
            completion_message = (
                "Runtime installed, but PyTorch did not detect CUDA. "
                "SAM will run on CPU until NVIDIA drivers/CUDA are available."
            )
        else:
            completion_message = (
                f"Runtime installed. SAM is ready on {device.upper()}."
            )
        log_plugin_runtime_install_event(plugin.id, completion_message)
        update_plugin_runtime_install_job(
            plugin.id,
            status="completed",
            step="completed",
            message=completion_message,
            error=None,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        store_plugin_runtime_state(
            plugin.id,
            status="ready",
            device=device,
            message=None,
        )
    except RuntimeError as exc:
        mark_plugin_runtime_install_failed(plugin.id, str(exc))
    except Exception as exc:  # pragma: no cover - defensive install guard
        mark_plugin_runtime_install_failed(
            plugin.id,
            f"Unexpected runtime installation failure: {exc}",
        )


def serialize_plugin_runtime(plugin: PluginDefinition):
    model_path = get_plugin_model_path(plugin)
    install_state = serialize_plugin_runtime_install(plugin)

    if plugin.id != "sam-3-1":
        return {
            "status": "ready",
            "device": None,
            "message": None,
            "supportsAutoAnnotate": False,
            "install": install_state,
        }

    model_installed = model_path.is_file()
    cached_runtime = load_cached_plugin_runtime_state(plugin)
    runtime_packages_available = has_sam_runtime_packages()

    if (
        cached_runtime is not None
        and cached_runtime["status"] == "ready"
        and not runtime_packages_available
    ):
        cached_runtime = None

    if install_state["status"] == "running":
        return {
            "status": "missing-runtime",
            "device": None,
            "message": install_state["message"] or "Preparing plugin runtime.",
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    if install_state["status"] == "failed":
        return {
            "status": "missing-runtime",
            "device": None,
            "message": install_state["error"] or "Plugin runtime installation failed.",
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    if not model_installed:
        return {
            "status": "missing-model",
            "device": cached_runtime["device"] if cached_runtime else None,
            "message": (
                "Runtime is installed. Download the model before using SAM auto-annotation."
                if (
                    install_state["status"] == "completed"
                    or (cached_runtime is not None and cached_runtime["status"] == "ready")
                    or runtime_packages_available
                )
                else "Download the model before using this plugin runtime."
            ),
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    if cached_runtime is not None:
        return {
            "status": cached_runtime["status"],
            "device": cached_runtime["device"],
            "message": cached_runtime["message"],
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    if install_state["status"] == "completed":
        return {
            "status": "ready",
            "device": install_state["resolvedProfile"],
            "message": None,
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    if not runtime_packages_available:
        runtime_error = (
            "SAM 3 runtime is not installed. Add torch, sam3 to the backend "
            "environment to enable auto-annotation."
        )
        store_plugin_runtime_state(
            plugin.id,
            status="missing-runtime",
            device=None,
            message=runtime_error,
        )
        return {
            "status": "missing-runtime",
            "device": None,
            "message": runtime_error,
            "supportsAutoAnnotate": True,
            "install": install_state,
        }

    return {
        "status": "missing-runtime",
        "device": None,
        "message": (
            "SAM runtime files were detected, but the runtime has not been verified yet. "
            "Use Install Runtime to verify and cache the environment."
        ),
        "supportsAutoAnnotate": True,
        "install": install_state,
    }


def resolve_sam_runtime_environment():
    ensure_supported_sam_runtime_host()
    ensure_supported_sam_runtime_platform()
    refresh_python_site_packages()
    missing_packages = []
    for module_name in ("torch", "sam3"):
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(module_name)

    if missing_packages:
        missing_list = ", ".join(missing_packages)
        raise RuntimeError(
            f"SAM 3 runtime is not installed. Add {missing_list} to the backend "
            "environment to enable auto-annotation.",
        )

    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        missing_dependency = exc.name or "unknown dependency"
        raise MissingSamRuntimeDependencyError(missing_dependency) from exc
    except Exception as exc:
        raise RuntimeError(
            f"SAM 3 runtime could not be imported: {exc}",
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official SAM 3 runtime currently requires an NVIDIA GPU with CUDA 12.6+. "
            "CPU-only runtime is not supported in this app.",
        )

    return {
        "torch": torch,
        "Sam3Processor": Sam3Processor,
        "build_sam3_image_model": build_sam3_image_model,
    }


def resolve_local_image_path(image_id: str, session_id: str | None = None):
    if session_id:
        session = LOCAL_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        image = session.images_by_id.get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        return image.full_path.resolve()

    image_path = LOCAL_IMAGE_PATHS.get(image_id)
    if image_path is None or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return image_path.resolve()


def get_plugin_runtime_cache_signature(plugin: PluginDefinition):
    model_path = get_plugin_model_path(plugin)
    stat_result = model_path.stat()
    return str(model_path), stat_result.st_mtime_ns


def get_sam_runtime_processor(plugin: PluginDefinition):
    model_path = get_plugin_model_path(plugin)
    if not model_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Download {plugin.name} model before running auto-annotation.",
        )

    try:
        runtime = resolve_sam_runtime_environment()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    cache_signature = get_plugin_runtime_cache_signature(plugin)

    with PLUGIN_RUNTIME_LOCK:
        cached = PLUGIN_RUNTIME_CACHE.get(plugin.id)
        if cached and cached.get("signature") == cache_signature:
            return cached["processor"], cached["torch"]

        torch_module = runtime["torch"]
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        build_sam3_image_model = runtime["build_sam3_image_model"]
        Sam3Processor = runtime["Sam3Processor"]

        try:
            try:
                model = build_sam3_image_model(
                    checkpoint_path=str(model_path),
                    load_from_HF=False,
                    device=device,
                    eval_mode=True,
                )
            except TypeError:
                model = build_sam3_image_model(
                    checkpoint_path=str(model_path),
                    load_from_HF=False,
                    device=device,
                )
                if hasattr(model, "eval"):
                    model.eval()

            processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
        except Exception as exc:
            LOGGER.exception("Failed to initialize SAM 3 runtime for %s", plugin.id)
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to initialize SAM 3 runtime ({exc.__class__.__name__}): {exc}"
                ),
            ) from exc

        cached = {
            "signature": cache_signature,
            "processor": processor,
            "torch": torch_module,
            "device": device,
            "loadedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        PLUGIN_RUNTIME_CACHE[plugin.id] = cached
        return processor, torch_module


def run_sam_auto_annotation(
    plugin: PluginDefinition,
    payload: PluginAutoAnnotateRequest,
):
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for SAM auto-annotation.")

    max_results = max(1, min(int(payload.maxResults), 64))
    score_threshold = max(0.0, min(float(payload.scoreThreshold), 1.0))
    image_path = resolve_local_image_path(payload.imageId, payload.sessionId)

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - Pillow is already required
        raise HTTPException(status_code=500, detail=f"Failed to load Pillow: {exc}") from exc

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    image_width, image_height = image.size
    crop_bounds = None
    crop_offset_x = 0
    crop_offset_y = 0

    if payload.mode == "selected-box":
        if payload.region is None:
            raise HTTPException(
                status_code=400,
                detail="Selected-box mode requires a source annotation region.",
            )

        crop_bounds = clamp_plugin_region_to_image(
            payload.region,
            image_width=image_width,
            image_height=image_height,
        )
        crop_offset_x, crop_offset_y, crop_right, crop_bottom = crop_bounds
        if crop_right - crop_offset_x < 2 or crop_bottom - crop_offset_y < 2:
            raise HTTPException(
                status_code=400,
                detail="Selected annotation region is too small for SAM auto-annotation.",
            )
        image = image.crop(crop_bounds)

    processor, torch_module = get_sam_runtime_processor(plugin)

    try:
        with torch_module.inference_mode():
            processor.set_confidence_threshold(score_threshold)
            if (
                getattr(torch_module, "cuda", None) is not None
                and torch_module.cuda.is_available()
            ):
                # SAM 3 image inference expects CUDA AMP, matching the upstream
                # examples and avoiding bf16/float32 mismatches inside the ViT.
                with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt=prompt)
            else:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)
    except Exception as exc:
        LOGGER.exception(
            "SAM 3 auto-annotation failed for plugin=%s image=%s prompt=%r",
            plugin.id,
            payload.imageId,
            prompt,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"SAM 3 auto-annotation failed ({exc.__class__.__name__}): {exc}"
            ),
        ) from exc

    boxes = coerce_sam_box_list(output.get("boxes"))
    scores = coerce_sam_score_list(output.get("scores"))
    annotations = []
    label = (payload.label or prompt).strip() or "object"
    crop_width, crop_height = image.size

    for index, raw_box in enumerate(boxes):
        box = normalize_sam_box(
            raw_box,
            image_width=crop_width,
            image_height=crop_height,
        )
        if box is None:
            continue

        score = scores[index] if index < len(scores) else 1.0
        if score < score_threshold:
            continue

        annotations.append(
            {
                "label": label,
                "score": round(score, 4),
                "x": round(box["x"] + crop_offset_x, 2),
                "y": round(box["y"] + crop_offset_y, 2),
                "width": round(box["width"], 2),
                "height": round(box["height"], 2),
            },
        )

    annotations.sort(key=lambda candidate: candidate["score"], reverse=True)
    annotations = annotations[:max_results]

    return {
        "pluginId": plugin.id,
        "mode": payload.mode,
        "prompt": prompt,
        "label": label,
        "annotationCount": len(annotations),
        "annotations": annotations,
    }


def clamp_plugin_region_to_image(
    region: PluginAnnotationRectPayload,
    *,
    image_width: int,
    image_height: int,
):
    left = max(0, min(int(region.x), image_width - 1 if image_width else 0))
    top = max(0, min(int(region.y), image_height - 1 if image_height else 0))
    right = max(left + 1, min(int(region.x + region.width), image_width))
    bottom = max(top + 1, min(int(region.y + region.height), image_height))
    return left, top, right, bottom


def coerce_sam_box_list(value: Any):
    raw_value = coerce_sam_output_value(value)
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple)):
        if raw_value and not isinstance(raw_value[0], (list, tuple)):
            return [raw_value]
        return [candidate for candidate in raw_value if isinstance(candidate, (list, tuple))]

    return []


def coerce_sam_score_list(value: Any):
    raw_value = coerce_sam_output_value(value)
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple)):
        normalized_scores = []
        for candidate in raw_value:
            try:
                normalized_scores.append(float(candidate))
            except (TypeError, ValueError):
                continue
        return normalized_scores

    try:
        return [float(raw_value)]
    except (TypeError, ValueError):
        return []


def coerce_sam_output_value(value: Any):
    if value is None:
        return None

    current = value
    for attribute_name in ("detach", "cpu"):
        attribute = getattr(current, attribute_name, None)
        if callable(attribute):
            current = attribute()

    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    return current


def normalize_sam_box(
    raw_box: Any,
    *,
    image_width: int,
    image_height: int,
):
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return None

    try:
        values = [float(raw_box[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None

    all_normalized = max(abs(value) for value in values) <= 1.5
    looks_like_xyxy = values[2] > values[0] and values[3] > values[1]

    if all_normalized and looks_like_xyxy:
        x1 = values[0] * image_width
        y1 = values[1] * image_height
        x2 = values[2] * image_width
        y2 = values[3] * image_height
    elif all_normalized:
        x1 = values[0] * image_width
        y1 = values[1] * image_height
        x2 = (values[0] + values[2]) * image_width
        y2 = (values[1] + values[3]) * image_height
    elif looks_like_xyxy:
        x1, y1, x2, y2 = values
    else:
        x1 = values[0]
        y1 = values[1]
        x2 = values[0] + values[2]
        y2 = values[1] + values[3]

    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(x1, min(x2, float(image_width)))
    y2 = max(y1, min(y2, float(image_height)))

    width = x2 - x1
    height = y2 - y1
    if width < 1.0 or height < 1.0:
        return None

    return {
        "x": x1,
        "y": y1,
        "width": width,
        "height": height,
    }


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


def delete_session_image_files(image: SessionImage):
    try:
        if image.full_path.exists():
            image.full_path.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image file: {image.full_path.name}",
        ) from exc

    if image.annotation_path is None:
        return

    try:
        if image.annotation_path.exists():
            image.annotation_path.unlink()
    except OSError:
        LOGGER.warning(
            "Failed to delete annotation sidecar for %s at %s",
            image.full_path,
            image.annotation_path,
            exc_info=True,
        )


def register_session_image(image: SessionImage):
    LOCAL_IMAGE_PATHS[image.id] = image.full_path


def register_session_images(images: list[SessionImage]):
    for image in images:
        register_session_image(image)


def remove_image_from_matching_sessions(root_path: Path, image_id: str):
    normalized_root_path = root_path.expanduser().resolve()

    for session in LOCAL_SESSIONS.values():
        if session.root_path != normalized_root_path:
            continue

        if image_id not in session.images_by_id:
            continue

        session.images = [
            current_image
            for current_image in session.images
            if current_image.id != image_id
        ]
        session.images_by_id.pop(image_id, None)

    LOCAL_IMAGE_PATHS.pop(image_id, None)


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
            return xml_path, "voc", count_pascal_voc_annotation_objects(xml_path)

        txt_path = annotation_base.parent / f"{annotation_base.name}.txt"
        if txt_path.is_file():
            return txt_path, "yolo", count_yolo_annotation_lines(txt_path)

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


def normalize_annotation_payloads(
    annotations: list[LocalAnnotationPayload],
) -> list[dict[str, Any]]:
    normalized_annotations: list[dict[str, Any]] = []

    for annotation in annotations:
        coordinates = (
            float(annotation.x),
            float(annotation.y),
            float(annotation.width),
            float(annotation.height),
        )
        if not all(math.isfinite(value) for value in coordinates):
            continue

        label = (annotation.label or "object").strip() or "object"
        normalized_annotations.append(
            {
                "id": uuid4().hex,
                "label": label,
                "difficult": bool(annotation.difficult),
                "x": max(coordinates[0], 0.0),
                "y": max(coordinates[1], 0.0),
                "width": max(coordinates[2], 0.0),
                "height": max(coordinates[3], 0.0),
            }
        )

    return normalized_annotations


def save_image_annotations(
    session: LocalSession,
    image: SessionImage,
    annotations: list[dict[str, Any]],
    *,
    preferred_classes: list[str],
):
    annotation_format = image.annotation_format or infer_session_annotation_format(session)
    annotation_path = resolve_annotation_output_path(session, image, annotation_format)

    if annotation_format == "voc":
        write_pascal_voc_annotations(annotation_path, image, annotations)
        class_source_path = None
        class_source_mtime_ns = None
    else:
        existing_class_source = (
            load_yolo_classes(image.annotation_path.parent, session.root_path)
            if image.annotation_path is not None
            else YoloClassSource(names=[])
        )
        class_names = build_yolo_class_list(
            annotations,
            preferred_classes=preferred_classes,
            existing_class_source=existing_class_source,
        )
        classes_path = resolve_yolo_classes_output_path(
            session,
            annotation_path,
            existing_class_source,
        )
        write_yolo_classes(classes_path, class_names)
        write_yolo_annotations(annotation_path, image.full_path, annotations, class_names)
        class_source_path = classes_path
        class_source_mtime_ns = classes_path.stat().st_mtime_ns

    annotation_mtime_ns = annotation_path.stat().st_mtime_ns
    image.annotation_path = annotation_path
    image.annotation_format = annotation_format
    image.annotation_count = len(annotations)

    CACHE_STORE.save_annotation_cache(
        annotation_path=annotation_path,
        annotation_format=annotation_format,
        annotation_mtime_ns=annotation_mtime_ns,
        image_path=image.full_path,
        image_mtime_ns=image.mtime_ns,
        class_source_path=class_source_path,
        class_source_mtime_ns=class_source_mtime_ns,
        payload=annotations,
    )
    CACHE_STORE.save_dataset_manifest(
        session.root_path,
        session.label,
        serialize_manifest_images(session.images),
    )


def infer_session_annotation_format(session: LocalSession):
    format_counts = {"yolo": 0, "voc": 0}

    for image in session.images:
        if image.annotation_format in format_counts:
            format_counts[image.annotation_format] += 1

    if format_counts["voc"] > format_counts["yolo"]:
        return "voc"

    return "yolo"


def infer_annotation_base_index(session: LocalSession, annotation_format: str):
    extension = ".xml" if annotation_format == "voc" else ".txt"
    index_counts: dict[int, int] = {}

    for image in session.images:
        if image.annotation_format != annotation_format or image.annotation_path is None:
            continue

        expected_path = image.annotation_path.with_suffix(extension)
        for index, annotation_base in enumerate(
            iter_annotation_bases(image.full_path, session.root_path)
        ):
            candidate_path = annotation_base.parent / f"{annotation_base.name}{extension}"
            if candidate_path == expected_path:
                index_counts[index] = index_counts.get(index, 0) + 1
                break

    if not index_counts:
        return None

    return min(
        index_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]


def resolve_annotation_output_path(
    session: LocalSession,
    image: SessionImage,
    annotation_format: str,
):
    if image.annotation_path is not None and image.annotation_format == annotation_format:
        return image.annotation_path

    extension = ".xml" if annotation_format == "voc" else ".txt"
    annotation_bases = list(iter_annotation_bases(image.full_path, session.root_path))
    preferred_index = infer_annotation_base_index(session, annotation_format)

    if preferred_index is None and annotation_format == "yolo":
        preferred_index = next(
            (
                index
                for index, annotation_base in enumerate(annotation_bases)
                if annotation_base.parent.name.lower() in LABEL_DIRECTORY_NAMES
            ),
            None,
        )

    if preferred_index is None or preferred_index >= len(annotation_bases):
        preferred_index = 0

    annotation_base = annotation_bases[preferred_index]
    return annotation_base.parent / f"{annotation_base.name}{extension}"


def resolve_yolo_classes_output_path(
    session: LocalSession,
    annotation_path: Path,
    existing_class_source: YoloClassSource,
):
    if (
        existing_class_source.source_path is not None
        and existing_class_source.source_path != PREDEFINED_CLASSES_FILE
    ):
        return existing_class_source.source_path

    annotation_parent = annotation_path.parent
    if annotation_parent.name.lower() in LABEL_DIRECTORY_NAMES:
        return annotation_parent / "classes.txt"

    if session.root_path.name.lower() in IMAGE_DIRECTORY_NAMES:
        return session.root_path.parent / "classes.txt"

    return session.root_path / "classes.txt"


def build_yolo_class_list(
    annotations: list[dict[str, Any]],
    *,
    preferred_classes: list[str],
    existing_class_source: YoloClassSource,
):
    class_names: list[str] = []
    seen: set[str] = set()

    def add_class(label: str):
        normalized = label.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        class_names.append(normalized)

    if (
        existing_class_source.source_path is not None
        and existing_class_source.source_path != PREDEFINED_CLASSES_FILE
    ):
        for label in existing_class_source.names:
            add_class(label)

        for label in preferred_classes:
            add_class(label)
    else:
        for label in preferred_classes:
            add_class(label)

    for annotation in annotations:
        add_class(str(annotation.get("label") or "object"))

    return class_names


def write_pascal_voc_annotations(
    annotation_path: Path,
    image: SessionImage,
    annotations: list[dict[str, Any]],
):
    image_width, image_height = read_image_size(image.full_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<annotation verified="no">',
        '\t<folder>browser</folder>',
        f'\t<filename>{html.escape(image.name)}</filename>',
        '\t<source>',
        '\t\t<database>Unknown</database>',
        '\t</source>',
        '\t<size>',
        f'\t\t<width>{image_width}</width>',
        f'\t\t<height>{image_height}</height>',
        '\t\t<depth>3</depth>',
        '\t</size>',
        '\t<segmented>0</segmented>',
    ]

    for annotation in annotations:
        box = annotation_to_pascal_voc_box(annotation, image_width, image_height)
        lines.extend(
            [
                '\t<object>',
                f'\t\t<name>{html.escape(str(annotation["label"]))}</name>',
                '\t\t<pose>Unspecified</pose>',
                f'\t\t<truncated>{1 if box["truncated"] else 0}</truncated>',
                f'\t\t<difficult>{1 if annotation.get("difficult") else 0}</difficult>',
                '\t\t<bndbox>',
                f'\t\t\t<xmin>{box["x_min"]}</xmin>',
                f'\t\t\t<ymin>{box["y_min"]}</ymin>',
                f'\t\t\t<xmax>{box["x_max"]}</xmax>',
                f'\t\t\t<ymax>{box["y_max"]}</ymax>',
                '\t\t</bndbox>',
                '\t</object>',
            ]
        )

    lines.append('</annotation>')
    annotation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def annotation_to_pascal_voc_box(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
):
    x_min = clamp_int(round(float(annotation["x"])), 1, image_width)
    y_min = clamp_int(round(float(annotation["y"])), 1, image_height)
    x_max = clamp_int(
        round(float(annotation["x"]) + float(annotation["width"])),
        1,
        image_width,
    )
    y_max = clamp_int(
        round(float(annotation["y"]) + float(annotation["height"])),
        1,
        image_height,
    )

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "truncated": (
            x_min == 1
            or y_min == 1
            or x_max == image_width
            or y_max == image_height
        ),
    }


def write_yolo_classes(classes_path: Path, class_names: list[str]):
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(class_names)
    classes_path.write_text(f"{contents}\n" if contents else "", encoding="utf-8")


def write_yolo_annotations(
    annotation_path: Path,
    image_path: Path,
    annotations: list[dict[str, Any]],
    class_names: list[str],
):
    image_width, image_height = read_image_size(image_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for annotation in annotations:
        label = str(annotation.get("label") or "object").strip() or "object"
        if label not in class_names:
            continue

        class_index = class_names.index(label)
        x_center = (float(annotation["x"]) + float(annotation["width"]) / 2) / image_width
        y_center = (float(annotation["y"]) + float(annotation["height"]) / 2) / image_height
        width = float(annotation["width"]) / image_width
        height = float(annotation["height"]) / image_height
        lines.append(
            " ".join(
                [
                    str(class_index),
                    f"{x_center:.6f}",
                    f"{y_center:.6f}",
                    f"{width:.6f}",
                    f"{height:.6f}",
                ]
            )
        )

    contents = "\n".join(lines)
    annotation_path.write_text(f"{contents}\n" if contents else "", encoding="utf-8")


def clamp_int(value: int, minimum: int, maximum: int):
    return min(max(value, minimum), maximum)


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


def count_pascal_voc_annotation_objects(annotation_path: Path):
    try:
        root = ElementTree.parse(annotation_path).getroot()
    except Exception:
        return 0

    return len(root.findall("object"))


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


def count_yolo_annotation_lines(annotation_path: Path):
    try:
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0

    annotation_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        try:
            int(parts[0])
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except ValueError:
            continue

        annotation_count += 1

    return annotation_count


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
