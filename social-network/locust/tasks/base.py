"""Base TaskSet for socialNetwork. socialNetwork's wrk2-api is form-encoded and
session-less (no auth tokens), so this is much thinner than train-ticket's base."""
from __future__ import annotations

from typing import Optional, Tuple
from locust import TaskSet


class BaseTaskSet(TaskSet):
    """Common helpers for all SN task sets.

    Adds light response checking: any HTTP status >= 400 marks failure.
    Locust's catch_response is used so individual endpoints can also override
    success/failure semantics (e.g. compose returns 200 on success but the
    body may have an "error" field — we treat HTTP 200 as success for now since
    DSB's services don't follow a uniform business-status envelope).
    """

    def _check(self, response, name: str) -> bool:
        if response.status_code >= 400:
            response.failure(f"HTTP {response.status_code} on {name}: {response.text[:120]}")
            return False
        response.success()
        return True

    def post_form(self, url: str, data: dict, name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        with self.client.post(url, data=data, name=name or url, catch_response=True) as r:
            ok = self._check(r, name or url)
            return ok, r.text if ok else None

    def get_qs(self, url: str, params: dict, name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        with self.client.get(url, params=params, name=name or url, catch_response=True) as r:
            ok = self._check(r, name or url)
            return ok, r.text if ok else None
