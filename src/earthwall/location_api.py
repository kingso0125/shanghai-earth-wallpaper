from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .location import Location, LocationStore
from .publisher import Publisher


LOGGER = logging.getLogger("earthwall.location_api")


class LocationApplication:
    def __init__(self, store: LocationStore, publisher: Publisher, token: str):
        if len(token) < 24:
            raise ValueError("location API token must contain at least 24 characters")
        self.store = store
        self.publisher = publisher
        self.token = token
        self._guard = threading.Lock()
        self._worker: threading.Thread | None = None
        self._pending: Location | None = None
        self._error: str | None = None

    def _queue(self, location: Location) -> None:
        with self._guard:
            self._pending = location
            if self._worker is None:
                self._worker = threading.Thread(target=self._render_pending, daemon=True)
                self._worker.start()

    def _render_pending(self) -> None:
        while True:
            with self._guard:
                location, self._pending = self._pending, None
                if location is None:
                    self._worker = None
                    return
            try:
                self.publisher.publish(location)
                self._error = None
            except Exception:
                LOGGER.exception("location render failed; keeping last good wallpaper")
                self._error = "render failed; last good wallpaper retained"

    def status(self, authorization: str) -> tuple[int, dict]:
        if not hmac.compare_digest(authorization, f"Bearer {self.token}"):
            return HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
        response = {"rendering": self._worker is not None, "error": self._error}
        try:
            current = (self.publisher.root / "current").resolve(strict=True)
            manifest = json.loads((current / "manifest.json").read_text())
            response.update({
                "target": manifest["target"],
                "observation_utc": manifest["observation_utc"],
                "version": current.name,
                "lock_path": f"/earthwall/releases/{current.name}/lock.jpg",
                "home_path": f"/earthwall/releases/{current.name}/home.jpg",
            })
        except (AttributeError, FileNotFoundError, KeyError, ValueError):
            response["version"] = None
        return HTTPStatus.OK, response

    def update(self, authorization: str, payload: dict) -> tuple[int, dict]:
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(authorization, expected):
            return HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
        try:
            accuracy = float(payload.get("accuracy", 0.0))
            if not math.isfinite(accuracy) or accuracy < 0 or accuracy > 20_000:
                raise ValueError("location accuracy is outside the accepted range")
            candidate = Location(
                float(payload["latitude"]),
                float(payload["longitude"]),
                str(payload.get("name") or "Current location").strip(),
            )
            with self._guard:
                location, distance, changed = self.store.update(candidate)
            if changed or self._error is not None:
                self._queue(location)
            return (HTTPStatus.ACCEPTED if changed else HTTPStatus.OK), {
                "changed": changed,
                "distance_km": round(distance, 1),
                "threshold_km": self.store.threshold_km,
                "target": {
                    "name": location.name,
                    "latitude": round(location.latitude, 4),
                    "longitude": round(location.longitude, 4),
                },
                "rendering": self._worker is not None,
                "version": "pending" if changed else "current",
            }
        except (KeyError, TypeError, ValueError) as error:
            return HTTPStatus.BAD_REQUEST, {"error": str(error)}


def handler_for(application: LocationApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EarthwallLocation/1"

        def do_GET(self):
            if self.path == "/status":
                status, response = application.status(self.headers.get("Authorization", ""))
                self._json(status, response)
                return
            if self.path != "/health":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.OK, {"status": "ok"})

        def do_POST(self):
            if self.path != "/location":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 4096:
                    raise ValueError("request body must contain 1 to 4096 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            status, response = application.update(
                self.headers.get("Authorization", ""), payload
            )
            self._json(status, response)

        def log_message(self, message, *args):
            LOGGER.info("%s %s", self.client_address[0], message % args)

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Location-aware Earth wallpaper service")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8131)
    result.add_argument("--state", type=Path, default=Path("/var/lib/earthwall/location.json"))
    result.add_argument("--root", type=Path, default=Path("/srv/earthwall"))
    result.add_argument("--cache", type=Path, default=Path("/var/cache/earthwall"))
    result.add_argument("--lock", type=Path, default=Path("/var/lib/earthwall/render.lock"))
    result.add_argument("--token-file", type=Path, default=Path("/etc/earthwall/location-token"))
    result.add_argument("--threshold-km", type=float, default=80.0)
    result.add_argument("--publish-once", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    store = LocationStore(
        args.state,
        args.threshold_km,
        apply_travel_override=True,
    )
    publisher = Publisher(args.root, args.cache, args.lock)
    if args.publish_once:
        publisher.publish(store.load())
        return 0

    token = args.token_file.read_text(encoding="utf-8").strip()
    application = LocationApplication(store, publisher, token)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(application))
    LOGGER.info("location API listening on %s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
