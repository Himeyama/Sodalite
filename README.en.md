<h1><img src="docs/icon.png" height="24" />&nbsp;Sodalite</h1>

[日本語](README.md) | English

A Stable Diffusion image-generation desktop app, built from scratch as an original implementation.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/e1151f70-408a-4a58-9549-4c01de492100" />

- **Frontend**: WinUI3 (.NET 9 / Windows App SDK)
- **Backend**: Python 3.13+ / FastAPI / diffusers (managed with uv)
- **Communication**: The frontend launches the backend as a local subprocess and talks to it over HTTP.
- The safety checker is disabled, so 🔞NSFW image generation is possible. Users are responsible for how generated content is used or shared, and must comply with the license of the model in use and the laws of their jurisdiction.

## Download and install

- GPU: 12 GiB or more of VRAM is recommended. (It can also run with 10 GiB of VRAM.)
- You need the Windows build of the `uv` command installed.  
  https://docs.astral.sh/uv/#installation

- Download the latest EXE from [Releases](https://github.com/Himeyama/Sodalite/releases).  
  Because this is an individually developed app, the browser will temporarily block it, but you can keep it via "⋯" → "Save" → the "∨" next to the "Delete" button → "Keep".

  <img src="docs\download-save-app.png" width="200" />

  For security, the installer is built automatically from [Actions](https://github.com/Himeyama/Sodalite/actions).

## Directory layout

```
Sodalite/
├── backend/           # Python backend (managed with uv, FastAPI + diffusers)
│   ├── src/sodalite_backend/
│   │   ├── main.py            # FastAPI entry point
│   │   ├── config.py          # Startup config (port, model id)
│   │   ├── api/                # REST API endpoints
│   │   ├── schemas/            # Pydantic request/response models
│   │   ├── inference/           # diffusers pipeline management, samplers
│   │   └── imaging/            # PNG metadata embedding, image saving
│   └── tests/
├── frontend/Sodalite/    # WinUI3 frontend
│   ├── MainWindow.xaml(.cs)    # Backend startup, navigation
│   ├── Views/GenerationPage    # Prompt input, generation, image display
│   ├── ViewModels/              # GenerationViewModel
│   └── Services/                # BackendProcessManager, BackendApiClient
├── docs/               # Documentation such as setup notes
├── skills/             # Development conventions (winui3-app, python-coding)
├── run.ps1             # App launch script (root)
└── CLAUDE.md
```

## Setup

### Prerequisites

- Windows 10 22H2 or later / Windows 11
- .NET 9 SDK
- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU (CUDA-capable, 8GB+ VRAM recommended; runs on CPU only, but slowly)

### First-time setup

```powershell
# Install backend dependencies
cd backend
uv sync
```

On first launch, the backend automatically downloads an image-generation model from Hugging Face (default: `stabilityai/sd-turbo`).

### Launch

```powershell
# Run from the root: syncs the backend, builds the frontend, and launches, all in one step
./run.ps1
```

When the app starts, the WinUI3 process automatically launches the Python backend as a child process (it dynamically detects a free port and waits until the health check passes). Closing the app reliably terminates the backend child process as well.

### Smoke-testing the backend on its own

```powershell
cd backend
./run.ps1
# In a separate terminal
curl http://localhost:8000/api/v1/health
```

## Development

- Python backend coding conventions: [`skills/python-coding/SKILL.md`](skills/python-coding/SKILL.md)
- WinUI3 frontend coding conventions: [`skills/winui3-app/SKILL.md`](skills/winui3-app/SKILL.md)

```powershell
# Backend lint/format/test
cd backend
uv run ruff check --fix .
uv run ruff format .
uv run pytest

# Frontend build
cd frontend/Sodalite
dotnet build -c Debug
```

## Distribution (installer)

You can build an NSIS installer (`Sodalite-Setup-<version>.exe`) for end users.

### Prerequisites

- **Build side**: .NET 9 SDK, [NSIS](https://nsis.sourceforge.io/) (`makensis`)
- **End-user side**: [uv](https://docs.astral.sh/uv/) must be installed.
  Python 3.13 is fetched automatically by uv during first-time setup, so no separate Python installation is required.

### Building the installer

```powershell
# Run from the root (publish → stage backend → NSIS compile, all in one)
./installer/build-installer.ps1 -Version 1.0.0
# or
make installer
```

This produces `installer/dist/Sodalite-Setup-<version>.exe`.

### First launch after installation

The installer places `app\` (frontend) and `backend\` (Python source) under
`%LOCALAPPDATA%\Programs\Sodalite\` (a per-user install requiring no administrator
privileges). The `.venv` is not bundled; **on first launch the frontend runs `uv sync`**
to create the virtual environment and install the dependencies (several GB for torch and
friends, taking a few minutes to tens of minutes).

- The setup result is recorded in `%LOCALAPPDATA%\Sodalite\.venv-ready`
  (it stores the hash of `uv.lock`). The marker is written only on success; if setup fails,
  **re-setup runs automatically on the next launch**. It also re-syncs when an app update
  changes the dependencies.
- If uv is not found, a message prompting you to install uv is shown when the app starts.

## API overview

The backend provides a purpose-built REST API (`/api/v1/*`).

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Startup check, currently loaded model, device info |
| GET | `/api/v1/samplers` | List of available samplers |
| POST | `/api/v1/generations/text-to-image` | txt2img generation |

See `backend/src/sodalite_backend/api/` for details.

## Disclaimer

- This app is a tool for running image-generation models. The author bears no responsibility whatsoever for the content, use, or distribution of generated images. Use of generated content is entirely at the user's own responsibility.
- Generated images are subject to the license of the model used (terms of use, prohibited uses, etc., as stated in the model card). Whether commercial use, redistribution, or NSFW content is permitted varies by model, so always check the license terms of the model you use before generating.
- Generated images may resemble or infringe upon existing copyrighted works, trademarks, or likeness rights, whether via the prompt or the training data. When publishing, distributing, or commercially using generated content, users are responsible for confirming that it does not infringe the rights of third parties.
- Generating images that imitate or evoke existing characters or other IP (intellectual property) from anime, games, movies, and the like can also infringe copyright or trademark rights. In particular, commercial use, redistribution, or sexual depictions not authorized by the rights holder of that IP may constitute infringement or a violation of platform terms, so users are responsible for making their own judgment.
- Generating or using images depicting real individuals (impersonation, defamation, unauthorized use in a sexual context, etc.) may violate the laws of your jurisdiction or the terms of the platforms you use. Compliance with applicable laws and terms of service is the user's responsibility.
- Generating, possessing, or distributing content prohibited by law is strictly forbidden.
