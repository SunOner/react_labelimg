# labelImg Next

`labelImg Next` is the new browser-based version of `labelImg`.

## Start

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