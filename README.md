# labelImg Next

Browser-based image and video annotation. Supports YOLO and Pascal VOC.
App: [127.0.0.1:8000](http://127.0.0.1:8000).

## Local installation

Requires Python 3.12+ and Node.js 22.

### Windows

Double-click `start.bat`, or run:

```powershell
.\start.bat
```

`setup.bat` prepares dependencies without launching;
`install_sam.bat` installs or repairs SAM.

### Ubuntu

```bash
sudo apt install python3-venv python3-tk
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash ./build.sh
.venv/bin/python main.py
```

## Docker

Copy `.env.example` to `.env` and set `LABELIMG_DATASETS_DIR` to your dataset folder.

```bash
# Published image
docker compose -f docker-compose.ghcr.yml up
# Or build from source
docker compose up --build
```

Open `/datasets` through **File → Backend path**.

## Video

Open a dataset, then **Video**. Save annotated frames to the dataset;
empty frames require **Save as background**.

Videos stay in their original folders; progress is stored in `data/video-metadata/`.
Reselect a browser-selected video to resume. Deleting it does not remove saved frames.

## SAM

Requires an NVIDIA GPU, a CUDA 12.6+ driver and Hugging Face model access.
Use **Plugins** to install the runtime and download weights.
Windows launchers prepare the runtime automatically.

## Development

With `.venv` activated (`npm.cmd` instead of `npm` in Windows PowerShell):

```bash
python main.py --reload
npm run dev --prefix frontend   # Separate terminal; http://127.0.0.1:5173
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
npm test --prefix frontend
```
