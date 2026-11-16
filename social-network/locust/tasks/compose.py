"""Compose-post traffic: write workload for the wrk2-api."""
from __future__ import annotations

from locust import task

from .base import BaseTaskSet
from utils.data_generator import random_user_id, random_username, random_post_text


class ComposePostTasks(BaseTaskSet):
    @task(2)
    def compose_post(self):
        uid = random_user_id()
        self.post_form(
            "/wrk2-api/post/compose",
            data={
                "username": random_username(uid),
                "user_id": str(uid),
                "text": random_post_text(),
                "media_ids": "[]",
                "media_types": "[]",
                "post_type": "0",
            },
            name="POST /wrk2-api/post/compose",
        )
