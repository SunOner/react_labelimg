# labelImg Next

`labelImg Next` is the new browser-based version of `labelImg`.

## Start

### Docker

Docker is the simplest way to run the app on Windows and Ubuntu. It builds the
React frontend, installs the FastAPI backend dependencies and serves everything
from one container.

Copy the example environment file and set the host folder that contains your
images:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
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

Build a release image manually:

```bash
docker build -t labelimg-next:0.1.0 .
```

Tagged GitHub releases such as `v0.1.0` are also built by the Docker workflow
and published to GitHub Container Registry as `ghcr.io/<owner>/<repo>:0.1.0`.

### Local Python/Node

On Windows, run the project from inside WSL (WSL2 recommended). Open a WSL
terminal, switch to the repository there, and run all commands below from the
Linux environment. On Linux, use the same steps directly.

Install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the frontend:

```bash
cd frontend
npm install
npm run build
```

Run the application from the repository root:

```bash
source .venv/bin/activate
python main.py
```

Open in your browser from WSL or Windows:

- app: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

## Development

Run backend:

```bash
source .venv/bin/activate
python main.py --reload
```
