"""Random data factories for socialNetwork load. Aligned with the DSB
init_social_graph.py seeding (socfb-Reed98, 962 users with user_id 0..961)."""
from __future__ import annotations

import os
import random
import string
from typing import List

                                                            
MAX_USER_ID = int(os.environ.get("SN_MAX_USER_ID", "961"))


def random_user_id() -> int:
    """Uniform random user_id within the seeded range."""
    return random.randint(0, MAX_USER_ID)


def random_username(uid: int | None = None) -> str:
    """DSB seeds usernames as 'username_<id>'. Generating new ones uses a random tail."""
    if uid is not None:
        return f"username_{uid}"
    tail = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"u_{tail}"


def random_password(uid: int | None = None) -> str:
    if uid is not None:
        return f"password_{uid}"
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


                                                                              
SAMPLE_WORDS = [
    "hello", "world", "social", "network", "post", "share", "rca",
    "dataset", "research", "training", "anomaly", "metric", "trace",
    "log", "service", "compose", "timeline", "graph", "follow", "user",
]


def random_text(min_words: int = 8, max_words: int = 20) -> str:
    n = random.randint(min_words, max_words)
    return " ".join(random.choices(SAMPLE_WORDS, k=n))


def random_mentions(count: int = 1) -> List[str]:
    """Random @-mentions targeting existing seeded users."""
    return [f"@username_{random_user_id()}" for _ in range(count)]


def random_url() -> str:
    return f"http://example.com/{''.join(random.choices(string.ascii_lowercase, k=8))}"


def random_post_text() -> str:
    parts = [random_text()]
    if random.random() < 0.5:
        parts.extend(random_mentions(random.randint(1, 2)))
    if random.random() < 0.3:
        parts.append(random_url())
    return " ".join(parts)
