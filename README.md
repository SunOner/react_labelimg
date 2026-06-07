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
