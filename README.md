# labelImg Next

`labelImg Next` is the new browser-based version of `labelImg`.

## Start

Install backend dependencies:

```bash
python -m venv .venv
".venv\Scripts\activate"
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
python main.py
```

Open:

- app: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

## Development

Run backend with reload:

```bash
python main.py --reload
```

Run frontend dev server in another terminal:

```bash
cd frontend
npm run dev
```
