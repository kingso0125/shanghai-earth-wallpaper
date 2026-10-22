import unittest
from http.client import IncompleteRead
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from earthwall.sources import _request


class NetworkTests(unittest.TestCase):
    def response(self):
        response = MagicMock()
        response.__enter__.return_value.read1.side_effect = [b"image", b""]
        return response

    @patch("earthwall.sources.time.sleep")
    def test_transient_tls_failure_retries(self, sleep):
        with patch("earthwall.sources.urllib.request.urlopen", side_effect=[URLError("TLS EOF"), self.response()]) as request:
            self.assertEqual(_request("https://example.test/image", timeout=10), b"image")
            self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()

    @patch("earthwall.sources.time.sleep")
    def test_missing_frame_does_not_retry_permanent_error(self, sleep):
        error = HTTPError("https://example.test/absent", 404, "Not found", {}, None)
        with patch("earthwall.sources.urllib.request.urlopen", side_effect=error) as request:
            with self.assertRaises(HTTPError):
                _request(error.url)
            self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    @patch("earthwall.sources.time.sleep")
    def test_retries_are_bounded(self, _sleep):
        with patch("earthwall.sources.urllib.request.urlopen", side_effect=URLError("offline")) as request:
            with self.assertRaises(URLError):
                _request("https://example.test/image")
            self.assertEqual(request.call_count, 3)

    @patch("earthwall.sources.time.sleep")
    def test_truncated_http_body_can_recover(self, _sleep):
        broken = MagicMock()
        broken.__enter__.return_value.read1.side_effect = IncompleteRead(b"partial")
        with patch("earthwall.sources.urllib.request.urlopen", side_effect=[broken, self.response()]):
            self.assertEqual(_request("https://example.test/image"), b"image")

    @patch("earthwall.sources.time.sleep")
    def test_exhausted_partial_reads_reach_existing_cache_fallback(self, _sleep):
        broken = MagicMock()
        broken.__enter__.return_value.read1.side_effect = IncompleteRead(b"partial")
        with patch("earthwall.sources.urllib.request.urlopen", return_value=broken):
            with self.assertRaises(OSError):
                _request("https://example.test/image")

    @patch("earthwall.sources.time.sleep")
    def test_truncated_content_length_is_rejected(self, _sleep):
        broken = [self.response() for _ in range(3)]
        for response in broken:
            response.__enter__.return_value.headers.get.return_value = "9"
        with patch("earthwall.sources.urllib.request.urlopen", side_effect=broken):
            with self.assertRaisesRegex(OSError, "truncated"):
                _request("https://example.test/image")

    @patch("earthwall.sources.time.monotonic", side_effect=[0, 0, 0, 11, 11])
    def test_slow_stream_cannot_bypass_download_budget(self, _time):
        with patch("earthwall.sources.urllib.request.urlopen", return_value=self.response()):
            with self.assertRaisesRegex(OSError, "time budget"):
                _request("https://example.test/image", timeout=10)
