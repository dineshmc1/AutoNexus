"""Build the static Auto Nexus Studio bundle consumed by Vercel."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "autonexus" / "web_static"
OUTPUT = ROOT / "vercel-dist"


def main() -> None:
    api_base = os.getenv("AUTONEXUS_PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if not api_base.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise SystemExit(
            "AUTONEXUS_PUBLIC_API_BASE_URL must be an HTTPS Railway URL."
        )

    shutil.rmtree(OUTPUT, ignore_errors=True)
    assets = OUTPUT / "assets"
    shutil.copytree(SOURCE, assets)
    shutil.copy2(SOURCE / "index.html", OUTPUT / "index.html")
    config = {
        "apiBaseUrl": api_base,
        "deployment": "vercel",
    }
    (assets / "config.js").write_text(
        "window.AUTO_NEXUS_CONFIG = Object.freeze("
        + json.dumps(config, separators=(",", ":"))
        + ");\n",
        encoding="utf-8",
    )
    print(f"Built Vercel frontend at {OUTPUT}")


if __name__ == "__main__":
    main()
