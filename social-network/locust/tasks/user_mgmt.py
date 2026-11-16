"""Low-frequency user management: register-user + follow."""
from __future__ import annotations

import itertools
import random
from locust import task

from .base import BaseTaskSet
from utils.data_generator import (
    random_user_id, random_username, random_password,
)


                                                                                
                                         
_uid_counter = itertools.count(100_000 + random.randint(0, 1_000_000))


class UserMgmtTasks(BaseTaskSet):
    @task(1)
    def register_user(self):
        uid = next(_uid_counter)
        self.post_form(
            "/wrk2-api/user/register",
            data={
                "first_name": "first",
                "last_name": "last",
                "username": random_username(uid),
                "password": random_password(uid),
                "user_id": str(uid),
            },
            name="POST /wrk2-api/user/register",
        )

    @task(1)
    def follow_user(self):
        a = random_user_id()
        b = random_user_id()
        while b == a:
            b = random_user_id()
        self.post_form(
            "/wrk2-api/user/follow",
            data={
                "user_name": random_username(a),
                "followee_name": random_username(b),
            },
            name="POST /wrk2-api/user/follow",
        )
