"""Runtime configuration for the backend server."""

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    host: str
    port: int
    model_id: str | None


def load_config() -> AppConfig:
    """Build config from CLI args, falling back to the SODALITE_PORT env var.

    `--webui` only affects the default host: browser access from other LAN
    devices needs a non-loopback bind, so an unspecified host becomes
    ``0.0.0.0`` in that mode. The WinUI3 frontend never passes ``--webui`` and
    keeps binding to loopback. An explicit ``--host`` always wins.

    `--model-id` has no built-in default here: the app resumes the last model
    the user selected (see [active_model_store]) in preference to this value.
    `main.create_app` falls back to a hardcoded default only when neither a
    saved model nor `--model-id` is available (a truly first run).
    """
    parser = argparse.ArgumentParser(prog="sodalite-backend")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=int(os.environ.get("SODALITE_PORT", "8000")))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--webui", action="store_true")
    args = parser.parse_args()
    host = args.host if args.host is not None else ("0.0.0.0" if args.webui else "127.0.0.1")
    return AppConfig(host=host, port=args.port, model_id=args.model_id)
