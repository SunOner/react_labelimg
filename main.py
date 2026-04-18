import argparse
import sys
from pathlib import Path


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    backend_dir = root_dir / "backend"

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing backend dependencies. Run `pip install -r requirements.txt` first."
        ) from exc

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    parser = argparse.ArgumentParser(
        description="Single-process launcher for the new labelImg web app"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload for backend development",
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(backend_dir)] if args.reload else None,
        app_dir=str(backend_dir),
    )


if __name__ == "__main__":
    main()
