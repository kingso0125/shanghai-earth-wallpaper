from __future__ import annotations

import argparse
import hmac
import json
import logging
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

    def update(self, authorization: str, payload: dict) -> tuple[int, dict]:
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(authorization, expected):
            return HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
        try:
            accuracy = float(payload.get("accuracy", 0.0))
            if accuracy < 0 or accuracy > 20_000:
                raise ValueError("location accuracy is outside the accepted range")
            candidate = Location(
                float(payload["latitude"]),
                float(payload["longitude"]),
                str(payload.get("name") or "Current location").strip(),
            )
            location, distance, changed = self.store.update(candidate)
            manifest = None
            if changed:
                manifest = self.publisher.publish(location)
            return HTTPStatus.OK, {
                "changed": changed,
                "distance_km": round(distance, 1),
                "threshold_km": self.store.threshold_km,
                "target": {
                    "name": location.name,
                    "latitude": round(location.latitude, 4),
                    "longitude": round(location.longitude, 4),
                },
                "version": (
                    manifest["artifacts"]["home"]["sha256"][:16]
                    if manifest is not None
                    else "current"
                ),
            }
        except (KeyError, TypeError, ValueError) as error:
            return HTTPStatus.BAD_REQUEST, {"error": str(error)}


def handler_for(application: LocationApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EarthwallLocation/1"

        def do_GET(self):
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
    store = LocationStore(args.state, args.threshold_km)
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
