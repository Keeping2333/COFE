"""Timeline read traffic — the dominant read workload."""
from __future__ import annotations

from locust import task

from .base import BaseTaskSet
from utils.data_generator import random_user_id


class TimelineTasks(BaseTaskSet):
    @task(5)
    def read_user_timeline(self):
        uid = random_user_id()
        self.get_qs(
            "/wrk2-api/user-timeline/read",
            params={"user_id": str(uid), "start": "0", "stop": "10"},
            name="GET /wrk2-api/user-timeline/read",
        )

    @task(5)
    def read_home_timeline(self):
        uid = random_user_id()
        self.get_qs(
            "/wrk2-api/home-timeline/read",
            params={"user_id": str(uid), "start": "0", "stop": "10"},
            name="GET /wrk2-api/home-timeline/read",
        )
