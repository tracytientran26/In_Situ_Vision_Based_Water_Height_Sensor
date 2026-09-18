#!/usr/bin/env python3
"""Download the newest USGS Rocky Branch webcam image.

The output filename uses the local computer time: YYYYMMDD_HHMMSS.jpg.
Designed to be called once every 30 minutes by Windows Task Scheduler or cron.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = "https://api.waterdata.usgs.gov/nims/v0"
SITE_ID = "02169506"  # Rocky Branch at Whaley St., Columbia, SC
USER_AGENT = "RockyBranchImageCollector/1.0"


def get_json(url: str, api_key: str | None) -> object:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    request = Request(url, headers=headers)
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def download(url: str, destination: Path, api_key: str | None) -> None:
    headers = {"User-Agent": USER_AGENT, "Accept": "image/jpeg"}
    if api_key:
        headers["X-Api-Key"] = api_key
    request = Request(url, headers=headers)
    temporary = destination.with_suffix(".part")
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as file:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise RuntimeError(f"Expected an image, received {content_type!r}.")
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def find_camera(api_key: str | None) -> dict[str, object]:
    query = urlencode({"siteId": SITE_ID})
    cameras = get_json(f"{API_BASE}/cameras?{query}", api_key)
    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError(f"No NIMS camera was returned for USGS site {SITE_ID}.")

    # Prefer a currently active camera when the API includes that metadata.
    active = [camera for camera in cameras if isinstance(camera, dict) and camera.get("active") is not False]
    camera = active[0] if active else cameras[0]
    if not isinstance(camera, dict):
        raise RuntimeError("Unexpected camera response from the NIMS API.")
    return camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\tranp\OneDrive\Desktop\ARTS lab\In_Situ_Vision_Based_Water_Height_Sensor\Images\training images"),
        help="Folder that will receive images (default: the project's training images folder).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_USGS_PAT"),
        help="Optional USGS Water Data API key. Defaults to API_USGS_PAT if set.",
    )
    args = parser.parse_args()

    try:
        camera = find_camera(args.api_key)
        camera_id = camera.get("camId")
        image_dir = camera.get("overlayDir") or camera.get("smallDir")
        if not isinstance(camera_id, str) or not isinstance(image_dir, str):
            raise RuntimeError("The camera response did not contain camId and an image directory.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = args.output_dir / f"{timestamp}.jpg"
        newest_image_url = image_dir.rstrip("/") + f"/{camera_id}_newest.jpg"
        download(newest_image_url, destination, args.api_key)
        print(destination)
        return 0
    except (HTTPError, URLError, OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        print("If the API responds with 403 or 429, obtain a free USGS API key and set API_USGS_PAT.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
