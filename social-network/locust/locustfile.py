"""locustfile.py — socialNetwork workload entry point.

User classes (weighted via env vars, same convention as train-ticket):
  SocialNetUser       — balanced mix (default profile)
  SocialNetQueryUser  — 100% timeline reads (query profile)
  SocialNetFlowUser   — 100% compose-post writes (flow profile)
  SocialNetFullUser   — full coverage incl. register/follow (full profile)

Env vars (set by run_experiment.py based on PROFILE_ENV in batch_experiment.py):
  SN_W_DEFAULT, SN_W_QUERY, SN_W_FLOW, SN_W_FULL  (integer weights)

Locust command line (constructed by run_experiment.py):
  locust -f locustfile.py --host=http://<sn-vm>:8080 --headless -u N -r R -t T \
         --csv=... --html=...
"""
from __future__ import annotations

import os

from locust import HttpUser, between

from tasks.compose import ComposePostTasks
from tasks.timeline import TimelineTasks
from tasks.user_mgmt import UserMgmtTasks


def _w(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class SocialNetUser(HttpUser):
    """Balanced mix: heavy on reads, light on writes — mirrors a typical SN workload."""
    weight = _w("SN_W_DEFAULT", 3)
    wait_time = between(1, 3)
    tasks = {
        TimelineTasks: 5,                      
        ComposePostTasks: 2,                 
    }


class SocialNetQueryUser(HttpUser):
    """Read-only profile — used during query-flavored experiments."""
    weight = _w("SN_W_QUERY", 0)
    wait_time = between(1, 2)
    tasks = {TimelineTasks: 1}


class SocialNetFlowUser(HttpUser):
    """Write-heavy profile — exposes post-storage / mongo write paths under stress."""
    weight = _w("SN_W_FLOW", 0)
    wait_time = between(1, 3)
    tasks = {ComposePostTasks: 1}


class SocialNetFullUser(HttpUser):
    """Full coverage including register/follow — used for OOD-service experiments."""
    weight = _w("SN_W_FULL", 0)
    wait_time = between(1, 3)
    tasks = {
        TimelineTasks: 3,
        ComposePostTasks: 2,
        UserMgmtTasks: 1,
    }
