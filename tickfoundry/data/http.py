"""The HTTP layer: key auth, retries on 429 / 503 / transport errors, resumable file download."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from . import errors

DEFAULT_BASE_URL = "https://tickfoundry.com"
ENV_KEY = "TICKFOUNDRY_API_KEY"
ENV_URL = "TICKFOUNDRY_URL"
USER_AGENT = "tickfoundry-python"


def _retry_after(resp: httpx.Response) -> float | None:
    v = resp.headers.get("retry-after")
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        return None


def _body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"error": resp.text[:500] or f"http {resp.status_code}"}


class Http:
    """One authenticated session against the TickFoundry API.

    ``sleep`` and ``transport`` are injectable so tests run without waiting
    or a network.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        max_wait: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        require_key: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get(ENV_KEY)
        if require_key and not self.api_key:
            raise errors.MissingApiKey(
                f"no API key: pass api_key=... or set {ENV_KEY} (keys are created on https://tickfoundry.com/account/api)"
            )
        self.base_url = (base_url or os.environ.get(ENV_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.sleep = sleep
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport,
                                    follow_redirects=False)

    def close(self) -> None:
        self._client.close()

    # -- JSON calls --------------------------------------------------------
    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._json("GET", path, params=params)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._json("POST", path, json=body)

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        resp = self._send(method, path, **kw)
        if resp.status_code >= 400:
            raise errors.from_response(resp.status_code, _body(resp), retry_after=_retry_after(resp))
        return _body(resp)

    def get_raw(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        """GET returning (status, body) without raising: the poll route answers 410 as data."""
        resp = self._send("GET", path, params=params)
        return resp.status_code, _body(resp)

    def _send(self, method: str, path: str, **kw: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                resp = self._client.request(method, path, **kw)
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise errors.TickFoundryError(f"network error talking to {self.base_url}: {e}") from e
                self.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if resp.status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                body = _body(resp) if resp.status_code == 429 else None
                # The day's download allowance does not come back in a retry window.
                if isinstance(body, dict) and body.get("code") == "daily_download_limit":
                    return resp
                wait = _retry_after(resp)
                wait = self._backoff(attempt) if wait is None else wait
                if wait > self.max_wait:
                    return resp
                self.sleep(wait)
                attempt += 1
                continue
            return resp

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(30.0, (2**attempt) * 0.5) + random.uniform(0, 0.25)

    # -- files ---------------------------------------------------------------
    def download(self, url: str, dest: Path, *, rebuild_deadline: float = 1800.0) -> Path:
        """Stream ``url`` to ``dest`` with resume (HTTP Range) and the file route's 202 "rebuilding".

        Writes ``dest.part`` and renames it when complete, so a crash never
        leaves a truncated file under the final name.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        started = time.monotonic()
        attempt = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Accept": "application/zip, application/octet-stream"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with self._client.stream("GET", url, headers=headers, timeout=None) as resp:
                    if resp.status_code == 202:
                        # Stored files are deleted after 7 days and rebuilt for free on the next fetch.
                        if time.monotonic() - started > rebuild_deadline:
                            raise errors.Timeout("the file is still being rebuilt; try again later", status=202)
                        self.sleep(_retry_after(resp) or 15.0)
                        continue
                    if resp.status_code == 416 and have:
                        part.rename(dest)  # we already had every byte
                        return dest
                    if resp.status_code in (429, 503) and attempt < self.max_retries:
                        self.sleep(min(self.max_wait, _retry_after(resp) or self._backoff(attempt)))
                        attempt += 1
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise errors.from_response(resp.status_code, _body(resp), retry_after=_retry_after(resp))
                    mode = "ab" if (have and resp.status_code == 206) else "wb"
                    with open(part, mode) as fh:
                        for chunk in resp.iter_bytes(1 << 20):
                            fh.write(chunk)
                part.rename(dest)
                return dest
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise errors.TickFoundryError(f"download interrupted: {e}") from e
                self.sleep(self._backoff(attempt))
                attempt += 1
