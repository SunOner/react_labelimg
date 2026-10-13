# labelImg Next

`labelImg Next` is the new browser-based version of `labelImg`.

## Start

### Docker: prebuilt image

Use the published GitHub Container Registry image when you only want to run the
app and do not need to rebuild it from source.

```bash
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
cp .env.example .env
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up
```

On Windows PowerShell:

```powershell
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
Copy-Item .env.example .env
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up
```

On Windows Command Prompt:

```bat
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
copy .env.example .env
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up
```

The default image tag is `main`:

```bash
ghcr.io/sunoner/react_labelimg:main
```

The Docker workflow publishes `main` and `latest` for default branch builds.
Tagged releases such as `v0.1.0` publish version tags such as `0.1.0` and
`0.1`.

### Docker: build from source

Docker is the simplest way to run the app on Windows and Ubuntu. It builds the
React frontend, installs the FastAPI backend dependencies and serves everything
from one container.

Clone the repository, copy the example environment file and set the host folder
that contains your images:

```bash
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
cp .env.example .env
```

On Windows PowerShell:

```powershell
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
Copy-Item .env.example .env
```

On Windows Command Prompt:

```bat
git clone https://github.com/SunOner/react_labelimg.git
cd react_labelimg
copy .env.example .env
```

Edit `.env` if your datasets are not in `./datasets`:

```env
LABELIMG_PORT=8000
LABELIMG_DATASETS_DIR=/absolute/path/to/your/datasets
```

Then run:

```bash
docker compose up --build
```

Open:

- app: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

In Docker, the dataset folder is mounted inside the container as `/datasets`.
Use `File -> Backend path`, enter `/datasets` or a file under it, then open it
as a directory or image.

Stop the app:

```bash
docker compose down
```

Build a local image manually:

```bash
docker build -t labelimg-next:local .
```

### Local Python/Node: Windows and Ubuntu

The app runs directly on Windows; WSL is optional. Install Python 3.12
and Node.js 22 (see `.nvmrc`). Run the following commands from the repository root.

#### Windows: batch launchers

Install Python 3.12 and Node.js 22, then double-click `start.bat` or run it
from Command Prompt/PowerShell. The first launch creates a Windows `.venv`,
installs backend dependencies, builds the frontend, and automatically installs
SAM dependencies when an NVIDIA GPU is detected. It then starts the app on
`http://127.0.0.1:8000`. Keep the console open while using the app.

```bat
start.bat
start.bat --port 8080
start.bat --reload
```

The launchers work from another working directory and from paths containing
spaces. Subsequent launches reuse backend/frontend dependencies and verify the
SAM runtime in a separate process before deciding whether installation is needed.

- `setup.bat`: install/update dependencies and rebuild without starting the app.
- `install_sam.bat`: install or repair SAM dependencies without building frontend;
  reports an error when an NVIDIA GPU is unavailable.
- `install_sam.bat -ForceSam`: reinstall an already verified SAM runtime.
- `setup.bat -SkipSam`: prepare only manual annotation.

The first SAM installation downloads CUDA-enabled PyTorch, matching Windows
Triton, the SAM source package, and missing Python dependencies. It
also installs the Microsoft Visual C++ runtime if its DLLs are missing, using
Microsoft's signed installer. Windows may display an elevation prompt for it.
Installation errors stop startup and remain visible in the console; retry
`start.bat` after fixing the reported problem. Model weights still require Hugging Face access
and are downloaded through the plugin dialog.

#### Windows PowerShell: manual setup

Use the Windows installations of Python and Node.js:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
.\.venv\Scripts\python.exe main.py
```

The execution policy override applies only to that build process. Activating the
Python environment is unnecessary. `build.ps1` uses `npm.cmd`, so it also works
when PowerShell blocks `npm.ps1`.

#### Ubuntu

Install Python's virtual environment and desktop file dialog support:

```bash
sudo apt install python3-venv python3-tk
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash ./build.sh
.venv/bin/python main.py
```

On Ubuntu releases whose default Python is older than 3.12, install Python 3.12
with its matching `venv` and `tk` packages and use `python3.12 -m venv .venv`.

Open in your browser:

- app: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Use `File -> Open image` / `Open directory` for the native file picker, or
`File -> Backend path` for a path on the computer running the backend:

- Windows: `C:\Users\you\Pictures\dataset` or `C:/Users/you/Pictures/dataset`;
  UNC paths such as `\\server\share\dataset` are also accepted.
- Ubuntu: `/home/you/datasets/example`.
- WSL: `/mnt/c/Users/you/Pictures/dataset`; Windows drive paths are translated
  to `/mnt/<drive>/...` by the Linux backend.

Headless Ubuntu requires `Backend path`; native dialogs need a graphical session.
In Docker on Windows, paths are container paths such as `/datasets`. Docker
Desktop must be configured to run Linux containers.

Do not reuse `.venv` or `frontend/node_modules` between Windows and Ubuntu/WSL.
Recreate `.venv` with the destination OS's Python and rebuild frontend dependencies
with `build.ps1 -Clean` or `bash ./build.sh --clean`. These clean commands remove
only `frontend/node_modules` and `frontend/dist`.

Both build scripts install dependencies with `npm ci` when `node_modules` is
missing. To reinstall dependencies, use `-Install` / `--install`; to check lint
as well, use `-Lint` / `--lint`. `-NoInstall` / `--no-install` skips installation.

### SAM auto-annotation

Manual annotation works without a GPU. The optional SAM 3.1 runtime needs
Python 3.12+, a compatible NVIDIA GPU, and a driver supporting CUDA 12.6+.
Use `Install Runtime` in the plugin dialog with the GPU/CUDA or automatic profile.
CPU-only SAM is unsupported.

The Windows batch launchers automatically prepare SAM; using the plugin's
`Install Runtime` button is also supported. On native Windows the installer adds
`triton-windows` matched to the installed
PyTorch minor version (currently PyTorch 2.7–2.10). Ubuntu uses PyTorch's Linux
Triton dependency. Windows also needs the Visual C++ Redistributable. See the
[SAM prerequisites](https://github.com/facebookresearch/sam3#installation) and
[Windows Triton requirements](https://github.com/triton-lang/triton-windows).
GPU compatibility depends on the selected PyTorch/Triton versions; runtime
installation verifies imports and CUDA availability, while inference also needs
the downloaded model and Hugging Face access.

## Development

Run the backend with automatic reload:

```powershell
.\.venv\Scripts\python.exe main.py --reload
```

```bash
.venv/bin/python main.py --reload
```

Run the frontend development server in a second terminal:

```powershell
npm.cmd run dev --prefix frontend
```

```bash
npm run dev --prefix frontend
```

Open `http://127.0.0.1:5173`; Vite proxies `/api` to port 8000. For another backend
port, set `LABELIMG_API_PROXY` before starting Vite.

The native CI workflow builds the frontend and tests backend file operations
on both `windows-latest` and `ubuntu-latest`. To run its backend tests locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

Frontend regression tests cover failed saves, concurrent saves, class indices,
annotation loading errors, session restoration, cache limits, and image resource
cleanup. Run them with `npm.cmd test --prefix frontend`
on Windows or `npm test --prefix frontend` on Ubuntu.

Duplicate search keeps at most 5,000 matching pairs per scan. At this limit the
scan stops and displays a partial-results notice; remove reviewed duplicates
and rescan, or increase the similarity threshold. Closing the search window
pauses browser polling while the bounded scan can finish on the server.
Result tiles and hover previews use images no larger than 512 pixels.

Browsing loads the original image only for the current frame and releases its
browser URL on navigation. Neighbor preloading fetches dimensions instead of
decoding original images. The annotation read cache keeps up to 24 images and
20,000 boxes, with exceptions for the current buffer, unsaved edits, and active
SAM requests. Class counts use compact summaries so cache eviction does not
remove class information.

Dataset opening runs cache validation and annotation refresh in a background
job, including when restoring the previous session. The sidebar shows the
validation/loading phase and its counters; cached images appear as their
annotation metadata is refreshed. Refreshing the page reuses a running job for
the same directory. Intermediate file lists are published at timed intervals
instead of after every 24 images, limiting repeated processing on large datasets.
Failed or timed-out restoration requests show an error and unlock opening.
