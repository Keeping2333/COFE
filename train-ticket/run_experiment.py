#!/usr/bin/env python3
"""
Production-Grade Experiment Runner for Multi-Modal RCA Dataset Generation
=========================================================================

This script orchestrates:
1. Locust workload on a worker VM (VM-TT)
2. One anomaly injection during the workload (4-layer architecture)
3. Export of multimodal telemetry (metrics/logs/traces) from Central VM
4. Saving everything into a structured run folder

Timeline ("sandwich" pattern):
  [locust_start] -> [injection_start] -> [injection_end] -> [locust_end]
     |<-- offset -->|<-- fault_duration -->|<-- recovery -->|

Usage:
  python3 run_experiment.py exp.json
  python3 run_experiment.py exp.json --dry-run
  python3 run_experiment.py exp.json --injection-start-time 180 --anomaly-duration 120

"""

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    print("ERROR: 'requests' module not found. Install with: pip3 install requests")
    sys.exit(1)

                                                                               
           
                                                                               

ANOMALY_CATALOG = {
                               
                                                            
                                                                              
                                                                                     
    "L1_01_node_cpu_pressure": {
        "filename": "L1_01_node_cpu_pressure.yaml",
        "layer": "L1",
        "root_cause_type": "cpu_pressure",
        "injection_tool": "stress-ng",
        "description": "Node CPU pressure at 95% load",
        "default_target_service": None,
    },
    "L1_02_disk_io_saturation": {
        "filename": "L1_02_disk_io_saturation.yaml",
        "layer": "L1",
        "root_cause_type": "disk_io_saturation",
        "injection_tool": "stress-ng",
        "description": "Disk I/O saturation",
        "default_target_service": None,
    },
    "L1_03_memory_pressure": {
        "filename": "L1_03_memory_pressure.yaml",
        "layer": "L1",
        "root_cause_type": "memory_pressure",
        "injection_tool": "stress-ng",
        "description": "Node memory pressure (OOM killer)",
        "default_target_service": None,
    },
                        
                                                                  
                                                                        
                                                                               
                                                                                    
    "L2_01_dns_failure": {
        "filename": "L2_01_dns_failure.yaml",
        "layer": "L2",
        "root_cause_type": "dns_failure",
        "injection_tool": "resolv.conf",
        "description": "DNS resolution failure for ts-order-service",
        "default_target_service": "ts-order-service",
    },
    "L2_02_network_latency": {
        "filename": "L2_02_network_latency.yaml",
        "layer": "L2",
        "root_cause_type": "network_latency",
        "injection_tool": "tc",
        "description": "200ms network latency on ts-travel-service egress",
        "default_target_service": "ts-travel-service",
    },
    "L2_03_packet_loss": {
        "filename": "L2_03_packet_loss.yaml",
        "layer": "L2",
        "root_cause_type": "packet_loss",
        "injection_tool": "tc",
        "description": "15% packet loss between ts-order-service and MySQL",
        "default_target_service": "ts-order-service",
    },
    "L2_04_connection_reset": {
        "filename": "L2_04_connection_reset.yaml",
        "layer": "L2",
        "root_cause_type": "connection_reset",
        "injection_tool": "iptables",
        "description": "TCP RST injection on ts-order-service inbound (OOD)",
        "default_target_service": "ts-order-service",
    },
                         
    "L3_01_pod_kill": {
        "filename": "L3_01_pod_kill.yaml",
        "layer": "L3",
        "root_cause_type": "pod_kill",
        "injection_tool": "kubectl",
        "description": "Pod kill / CrashLoopBackOff for ts-basic-service",
        "default_target_service": "ts-basic-service",
    },
    "L3_02_liveness_probe_failure": {
        "filename": "L3_02_liveness_probe_failure.yaml",
        "layer": "L3",
        "root_cause_type": "liveness_probe_failure",
        "injection_tool": "kubectl",
        "description": "Liveness probe failure for ts-order-service",
        "default_target_service": "ts-order-service",
    },
    "L3_03_resource_quota_exhaustion": {
        "filename": "L3_03_resource_quota_exhaustion.yaml",
        "layer": "L3",
        "root_cause_type": "resource_quota_exhaustion",
        "injection_tool": "kubectl",
        "description": "Resource quota exhaustion in tt namespace",
        "default_target_service": None,                                         
    },
    "L3_04_image_pull_error": {
        "filename": "L3_04_image_pull_error.yaml",
        "layer": "L3",
        "root_cause_type": "image_pull_error",
        "injection_tool": "kubectl",
        "description": "Image pull error via deployment tag patch (OOD)",
        "default_target_service": "ts-basic-service",
    },
                            
    "L4_01_jvm_memory_pressure": {
        "filename": "L4_01_jvm_memory_pressure.yaml",
        "layer": "L4",
        "root_cause_type": "jvm_memory_pressure",
        "injection_tool": "byteman",
        "description": "JVM heap memory pressure for ts-order-service",
        "default_target_service": "ts-order-service",
    },
    "L4_02_database_connection_exhaustion": {
        "filename": "L4_02_database_connection_exhaustion.yaml",
        "layer": "L4",
        "root_cause_type": "database_connection_exhaustion",
        "injection_tool": "mysql",
        "description": "HikariCP connection pool exhaustion",
        "default_target_service": "ts-order-service",
    },
    "L4_03_http_error_injection": {
        "filename": "L4_03_http_error_injection.yaml",
        "layer": "L4",
        "root_cause_type": "http_error_injection",
        "injection_tool": "iptables",
        "description": "HTTP 500 error injection for ts-travel-service",
        "default_target_service": "ts-travel-service",
    },
    "L4_04_slow_sql_injection": {
        "filename": "L4_04_slow_sql_injection.yaml",
        "layer": "L4",
        "root_cause_type": "slow_sql_injection",
        "injection_tool": "mysql",
        "description": "Slow SQL via LOCK TABLES WRITE on target's primary table (OOD)",
        "default_target_service": "ts-order-service",
    },
}

                                                                               
               
                                                                               

def setup_logging(run_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both console and file."""
    logger = logging.getLogger("experiment_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

                     
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

                                      
    if run_dir and run_dir.exists():
        file_handler = logging.FileHandler(run_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


                                                                               
                   
                                                                               

def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime) -> str:
    """Format datetime as UTC ISO8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to seconds.
    Supports formats: '5m', '300s', '300', '1h30m'
    """
    if not duration_str:
        return 0

    duration_str = duration_str.strip().lower()
    
                            
    if duration_str.isdigit():
        return int(duration_str)
    
    total_seconds = 0
    current_num = ""
    
    for char in duration_str:
        if char.isdigit():
            current_num += char
        elif char == 'h':
            total_seconds += int(current_num) * 3600 if current_num else 0
            current_num = ""
        elif char == 'm':
            total_seconds += int(current_num) * 60 if current_num else 0
            current_num = ""
        elif char == 's':
            total_seconds += int(current_num) if current_num else 0
            current_num = ""
    
                                                          
    if current_num:
        total_seconds += int(current_num)
    
    return total_seconds


def run_local_cmd(cmd: List[str], timeout: int = 60, capture: bool = True) -> Tuple[int, str, str]:
    """
    Run a local command and return (returncode, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


class SSHRunner:
    """SSH/SCP operations via subprocess (no paramiko)."""

    def __init__(self, host: str, user: str, key_path: str, logger: logging.Logger):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.logger = logger
        self._ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-i", key_path
        ]

    def run(self, command: str, timeout: int = 300, check: bool = False) -> Tuple[int, str, str]:
        """
        Execute command on remote host via SSH.
        Returns (returncode, stdout, stderr).
        """
        ssh_cmd = [
            "ssh",
            *self._ssh_opts,
            f"{self.user}@{self.host}",
            command
        ]
        self.logger.debug(f"SSH: {command}")
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if check and result.returncode != 0:
                self.logger.error(f"SSH command failed: {result.stderr}")
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"SSH command timed out after {timeout}s")
            return -1, "", f"Timeout after {timeout}s"
        except Exception as e:
            self.logger.error(f"SSH error: {e}")
            return -1, "", str(e)

    def run_background(self, command: str, output_file: str) -> Tuple[int, str, str]:
        """
        Run command in background on remote host with nohup.
        Output redirected to output_file on remote host.
        """
                                                                                
                                                                  
        escaped_command = command.replace("'", "'\\''")
        bg_command = f"nohup bash -c '{escaped_command}' > {output_file} 2>&1 & echo $!"
        return self.run(bg_command, timeout=30)

    def scp_from_remote(self, remote_path: str, local_path: str, timeout: int = 120) -> bool:
        """Copy file(s) from remote to local."""
        scp_cmd = [
            "scp",
            "-r",
            *self._ssh_opts,
            f"{self.user}@{self.host}:{remote_path}",
            local_path
        ]
        self.logger.debug(f"SCP from: {remote_path} -> {local_path}")
        
        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                self.logger.debug(f"SCP failed: {result.stderr.strip()}")
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"SCP error: {e}")
            return False

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on remote host."""
        rc, out, err = self.run(f"test -f {remote_path} && echo 'yes' || echo 'no'", timeout=10)
        return rc == 0 and "yes" in out

    def test_connection(self) -> bool:
        """Test SSH connectivity."""
        rc, out, err = self.run("echo 'ok'", timeout=15)
        return rc == 0 and "ok" in out


                                                                               
                     
                                                                               

class PrometheusExporter:
    """Export metrics from Prometheus via HTTP API."""

    def __init__(self, base_url: str, logger: logging.Logger):
        self.base_url = base_url.rstrip("/")
        self.logger = logger

    def health_check(self) -> bool:
        """Check Prometheus connectivity."""
        try:
            resp = requests.get(f"{self.base_url}/-/ready", timeout=10)
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Prometheus health check failed: {e}")
            return False

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: int = 15
    ) -> Optional[Dict]:
        """Execute a query_range request."""
        params = {
            "query": query,
            "start": utc_iso(start),
            "end": utc_iso(end),
            "step": f"{step}s"
        }
        url = f"{self.base_url}/api/v1/query_range"
        
        try:
            self.logger.debug(f"Prometheus query: {query[:80]}...")
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error(f"Prometheus query_range failed: {e}")
            return None

    def export_queries(
        self,
        queries: Dict[str, str],
        start: datetime,
        end: datetime,
        step: int,
        output_dir: Path
    ) -> Dict[str, str]:
        """Export multiple PromQL queries to files."""
        results = {}
        for name, query in queries.items():
                                                                            
            if name.startswith("_"):
                self.logger.debug(f"Skipping section header: {name}")
                continue
            
            self.logger.info(f"Exporting Prometheus metric: {name}")
            data = self.query_range(query, start, end, step)
            
            if data:
                filename = f"{name}.json"
                filepath = output_dir / filename
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                results[name] = str(filepath)
                
                                  
                if data.get("data", {}).get("result"):
                    series_count = len(data["data"]["result"])
                    self.logger.debug(f"  -> {series_count} time series exported")
            else:
                results[name] = "FAILED"
        
        return results


class OTelFileLogFetcher:
    """
    Fetch logs from TT VM's OTel Collector file exporter output.

    Replaces the previous Loki path. The Collector writes OTLP-JSON envelopes
    (one batch per line) to /var/log/otel/logs.jsonl on the worker node. We
    SSH there, filter the lines whose logRecords fall inside the experiment
    window, and SCP the result back as a single logs.jsonl file.
    """

    def __init__(
        self,
        ssh: "SSHRunner",
        remote_log_path: str,
        logger: logging.Logger,
    ):
        self.ssh = ssh
        self.remote_log_path = remote_log_path
        self.logger = logger

    def health_check(self) -> bool:
        """Verify the remote log file exists and is non-empty."""
        rc, out, err = self.ssh.run(
            f"sudo test -s {self.remote_log_path} && echo ok || echo missing",
            timeout=15,
        )
        ok = rc == 0 and "ok" in out
        if not ok:
            self.logger.error(
                f"OTel log file health check failed: path={self.remote_log_path} "
                f"rc={rc} out={out.strip()!r} err={err.strip()!r}"
            )
        return ok

    def fetch_window(
        self,
        start: datetime,
        end: datetime,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Filter remote logs.jsonl (and any rotated siblings) to records whose
        timestamp ∈ [start, end]; write the resulting JSONL locally.

        Returns a result dict with status + counts for meta.json.
        """
        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = int(end.timestamp() * 1_000_000_000)

                                                                           
                                                                         
                                            
        remote_tmp = f"/tmp/logs_window_{os.getpid()}.jsonl"

                                                                              
                                                                
        filter_script = (
            "import json, sys, glob\n"
            f"START={start_ns}\n"
            f"END={end_ns}\n"
            "kept = 0\n"
            "rec_total = 0\n"
            "rec_in = 0\n"
            f"paths = sorted(glob.glob('{self.remote_log_path}*'))\n"
            "for p in paths:\n"
            "    with open(p, 'r') as f:\n"
            "        for line in f:\n"
            "            line = line.rstrip('\\n')\n"
            "            if not line:\n"
            "                continue\n"
            "            try:\n"
            "                d = json.loads(line)\n"
            "            except Exception:\n"
            "                continue\n"
            "            keep = False\n"
            "            for rl in d.get('resourceLogs', []):\n"
            "                for sl in rl.get('scopeLogs', []):\n"
            "                    for lr in sl.get('logRecords', []):\n"
            "                        rec_total += 1\n"
            "                        t_str = lr.get('timeUnixNano') or lr.get('observedTimeUnixNano') or '0'\n"
            "                        try:\n"
            "                            t = int(t_str)\n"
            "                        except Exception:\n"
            "                            t = 0\n"
            "                        if START <= t <= END:\n"
            "                            keep = True\n"
            "                            rec_in += 1\n"
            "            if keep:\n"
            "                sys.stdout.write(line + '\\n')\n"
            "                kept += 1\n"
            "sys.stderr.write(f'kept_lines={kept} records_total={rec_total} records_in_window={rec_in}\\n')\n"
        )

                                                                            
                               
        import base64
        b64 = base64.b64encode(filter_script.encode()).decode()

                                                                             
                                            
        remote_cmd = (
            f"sudo bash -c \"python3 -c 'import base64,sys; "
            f"exec(base64.b64decode(\\\"{b64}\\\").decode())' > {remote_tmp}\""
        )
        self.logger.info(
            f"Filtering OTel logs on worker for window "
            f"[{utc_iso(start)}, {utc_iso(end)}]..."
        )
        rc, out, err = self.ssh.run(remote_cmd, timeout=300)
        if rc != 0:
            self.logger.error(
                f"Remote log filter failed: rc={rc} stderr={err.strip()[:300]!r}"
            )
            return {"status": "FAILED", "error": err.strip()[:500]}

                                                                  
        diag = (err or "").strip()
        self.logger.info(f"  remote filter: {diag}")

                                    
        local_tmp = str(output_path)
        ok = self.ssh.scp_from_remote(remote_tmp, local_tmp, timeout=300)
                                              
        self.ssh.run(f"sudo rm -f {remote_tmp}", timeout=15)

        if not ok:
            self.logger.error(f"SCP of filtered logs failed")
            return {"status": "FAILED", "error": "scp failed"}

                                     
        try:
            local_size = os.path.getsize(local_tmp)
        except OSError:
            local_size = 0

                                                                             
        line_count = 0
        record_count = 0
        try:
            with open(local_tmp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    line_count += 1
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    for rl in d.get("resourceLogs", []):
                        for sl in rl.get("scopeLogs", []):
                            record_count += len(sl.get("logRecords", []))
        except Exception as e:
            self.logger.warning(f"Could not introspect local logs file: {e}")

        self.logger.info(
            f"  -> {record_count} log records ({line_count} batches, "
            f"{local_size} bytes) -> {output_path}"
        )

        return {
            "status": "ok",
            "path": str(output_path),
            "remote_diagnostics": diag,
            "batches": line_count,
            "records": record_count,
            "bytes": local_size,
            "window_start_utc": utc_iso(start),
            "window_end_utc": utc_iso(end),
        }


class JaegerExporter:
    """Export traces from Jaeger via HTTP API."""

    def __init__(self, base_url: str, logger: logging.Logger):
        self.base_url = base_url.rstrip("/")
        self.logger = logger

    def health_check(self) -> bool:
        """Check Jaeger connectivity."""
        try:
            resp = requests.get(f"{self.base_url}/api/services", timeout=10)
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Jaeger health check failed: {e}")
            return False

    def get_services(self) -> List[str]:
        """Get list of services."""
        try:
            resp = requests.get(f"{self.base_url}/api/services", timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            self.logger.error(f"Jaeger get_services failed: {e}")
            return []

    def search_traces(
        self,
        service: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 10.0
    ) -> List[str]:
        """Search for trace IDs matching criteria. Retries on transient errors."""
        params = {
            "service": service,
            "start": int(start.timestamp() * 1e6),                
            "end": int(end.timestamp() * 1e6),
            "limit": limit
        }
        if operation:
            params["operation"] = operation
        if tags:
            params["tags"] = json.dumps(tags)

        url = f"{self.base_url}/api/traces"

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                trace_ids = [
                    t["traceID"] for t in data.get("data", []) if t.get("traceID")
                ]
                return trace_ids
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(
                        f"Jaeger search_traces failed (attempt {attempt}/{max_retries})"
                        f" for {service}: {e}. Retrying in {retry_delay_sec}s..."
                    )
                    time.sleep(retry_delay_sec)
                else:
                    self.logger.error(
                        f"Jaeger search_traces failed after {max_retries} attempts"
                        f" for {service}: {e}"
                    )
        return []

    def get_trace(
        self,
        trace_id: str,
        max_retries: int = 3,
        retry_delay_sec: float = 10.0
    ) -> Optional[Dict]:
        """Fetch a single trace by ID. Retries on transient errors."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    f"{self.base_url}/api/traces/{trace_id}", timeout=30
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < max_retries:
                    self.logger.debug(
                        f"Jaeger get_trace failed (attempt {attempt}/{max_retries})"
                        f" for {trace_id}: {e}. Retrying in {retry_delay_sec}s..."
                    )
                    time.sleep(retry_delay_sec)
                else:
                    self.logger.debug(
                        f"Jaeger get_trace failed after {max_retries} attempts"
                        f" for {trace_id}: {e}"
                    )
        return None

    def export_traces(
        self,
        services: List[str],
        start: datetime,
        end: datetime,
        limit_per_service: int,
        output_dir: Path,
        fetch_full_traces: bool = True,
        max_total_traces: int = 500,
        request_delay_sec: float = 0.2
    ) -> Dict[str, Any]:
        """Export traces for specified services."""
        import random

        results = {
            "services_queried": services,
            "trace_ids": [],
            "traces_exported": 0,
            "files": []
        }

        all_trace_ids = set()

                                             
        for service in services:
            self.logger.info(f"Searching traces for service: {service}")
            trace_ids = self.search_traces(
                service=service,
                start=start,
                end=end,
                limit=limit_per_service
            )
            self.logger.debug(f"  -> Found {len(trace_ids)} traces")
            all_trace_ids.update(trace_ids)
            if request_delay_sec > 0:
                time.sleep(request_delay_sec)

        results["trace_ids"] = list(all_trace_ids)
        self.logger.info(f"Total unique traces found: {len(all_trace_ids)}")

                               
        summary_file = output_dir / "trace_search_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "services": services,
                "start": utc_iso(start),
                "end": utc_iso(end),
                "trace_ids": results["trace_ids"],
                "total_traces": len(all_trace_ids)
            }, f, indent=2)
        results["files"].append(str(summary_file))

                                          
        if fetch_full_traces and all_trace_ids:
                                                                          
            fetch_ids = list(all_trace_ids)
            if max_total_traces > 0 and len(fetch_ids) > max_total_traces:
                fetch_ids = random.sample(fetch_ids, max_total_traces)
                self.logger.info(f"Subsampled to {max_total_traces} traces (from {len(all_trace_ids)} found)")

            self.logger.info(f"Fetching {len(fetch_ids)} full traces...")
            for trace_id in fetch_ids:
                trace_data = self.get_trace(trace_id)
                if trace_data:
                    trace_file = output_dir / f"{trace_id}.json"
                    with open(trace_file, "w") as f:
                        json.dump(trace_data, f, indent=2)
                    results["traces_exported"] += 1
                if request_delay_sec > 0:
                    time.sleep(request_delay_sec)

            self.logger.info(f"Exported {results['traces_exported']} traces to {output_dir}")

        return results


                                                                               
                   
                                                                               

class ExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        if getattr(args, "run_id", None):
            self.run_id = args.run_id
        else:
            self.run_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
                                       
        if args.injection_start_time is not None:
            self.config["anomaly"]["fault_offset_sec"] = args.injection_start_time
        if args.anomaly_duration is not None:
            self.config["anomaly"]["fault_duration_sec"] = args.anomaly_duration
        
                     
        self.base_dir = Path(config.get("output_base_dir", "exp_runs"))
        self.run_dir = self.base_dir / self.run_id
        
               
        self.meta: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.locust_pid: Optional[str] = None
        self.anomaly_applied = False
        self.cleanup_done = False
        self.logger: Optional[logging.Logger] = None
        self.ssh: Optional[SSHRunner] = None
        
                         
        self._shutdown_event = threading.Event()

    def setup(self) -> bool:
        """Initialize directories, logging, and connections."""
                                                                                  
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "traces").mkdir(exist_ok=True)
        
                       
        self.logger = setup_logging(self.run_dir, self.args.verbose)
        self.logger.info(f"=" * 60)
        self.logger.info(f"Experiment Runner - Run ID: {self.run_id}")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Output directory: {self.run_dir}")
        
                        
        ssh_config = self.config["ssh"]
        self.ssh = SSHRunner(
            host=ssh_config["host"],
            user=ssh_config["user"],
            key_path=os.path.expanduser(ssh_config["key_path"]),
            logger=self.logger
        )
        
                         
        self.meta = {
            "run_id": self.run_id,
            "created_at_utc": utc_iso(utc_now()),
            "config_file": self.args.config_file,
            "cli_args": vars(self.args),
            "timestamps": {},
            "locust": {},
            "anomaly": {},
            "export": {},
            "errors": [],
            "exit_status": "unknown"
        }
        
        return True

    def validate_connectivity(self) -> bool:
        """Validate all required connections before starting."""
        self.logger.info("Validating connectivity...")
        all_ok = True
        
                          
        self.logger.info("  Testing SSH connection to worker VM...")
        if not self.ssh.test_connection():
            self.logger.error("  [FAIL] Cannot connect to worker VM via SSH")
            self.errors.append("SSH connection to worker VM failed")
            all_ok = False
        else:
            self.logger.info("  [OK] SSH connection")
        
                    
        self.logger.info("  Testing Prometheus connection...")
        prom = PrometheusExporter(
            self.config["export"]["prometheus"]["base_url"],
            self.logger
        )
        if not prom.health_check():
            self.logger.error("  [FAIL] Prometheus not reachable")
            self.errors.append("Prometheus health check failed")
            all_ok = False
        else:
            self.logger.info("  [OK] Prometheus")
        
                                                           
        self.logger.info("  Testing OTel file log source on worker VM...")
        otel_log_path = self.config["export"]["otel_logs"]["remote_path"]
        log_fetcher = OTelFileLogFetcher(self.ssh, otel_log_path, self.logger)
        if not log_fetcher.health_check():
            self.logger.error(
                f"  [FAIL] OTel log file not present/empty at {otel_log_path}"
            )
            self.errors.append("OTel file log health check failed")
            all_ok = False
        else:
            self.logger.info("  [OK] OTel file log source")
        
                
        self.logger.info("  Testing Jaeger connection...")
        jaeger = JaegerExporter(
            self.config["export"]["jaeger"]["base_url"],
            self.logger
        )
        if not jaeger.health_check():
            self.logger.error("  [FAIL] Jaeger not reachable")
            self.errors.append("Jaeger health check failed")
            all_ok = False
        else:
            self.logger.info("  [OK] Jaeger")
        
                                              
        anomaly_id = self.config["anomaly"]["anomaly_id"]
        if anomaly_id not in ANOMALY_CATALOG:
            self.logger.error(f"  [FAIL] Unknown anomaly_id: {anomaly_id}")
            self.errors.append(f"Unknown anomaly_id: {anomaly_id}")
            all_ok = False
        else:
            anomaly_info = ANOMALY_CATALOG[anomaly_id]
            anomaly_dir = self.config["anomaly"]["anomaly_dir"]
            yaml_path = f"{anomaly_dir}/{anomaly_info['filename']}"
            rc, out, err = self.ssh.run(f"test -f {yaml_path} && echo 'exists'")
            if "exists" not in out:
                self.logger.error(f"  [FAIL] Anomaly YAML not found: {yaml_path}")
                self.errors.append(f"Anomaly YAML not found: {yaml_path}")
                all_ok = False
            else:
                self.logger.info(f"  [OK] Anomaly YAML exists: {anomaly_info['filename']}")

                                                                                           
        self.logger.info("  Checking HTTP application metrics in Prometheus...")
        try:
            prom_base = self.config["export"]["prometheus"]["base_url"]
            resp = __import__("requests").get(
                f"{prom_base}/api/v1/query",
                params={"query": "count(http_server_request_duration_seconds_count)"},
                timeout=10
            )
            result = resp.json().get("data", {}).get("result", [])
            if result and float(result[0].get("value", [0, "0"])[1]) > 0:
                self.logger.info(f"  [OK] HTTP metrics available ({result[0]['value'][1]} series)")
            else:
                self.logger.warning("  [WARN] HTTP application metrics not found in Prometheus.")
                self.logger.warning("         Ensure Train-Ticket services are running with OTel Java Agent.")
                self.logger.warning("         Run: kubectl -n tt get pods | grep -v '1/1.*Running'")
        except Exception as e:
            self.logger.warning(f"  [WARN] Could not verify HTTP metrics: {e}")

        return all_ok

    def build_locust_command(self, output_dir: str) -> str:
        """Build the Locust CLI command from config."""
        locust_config = self.config["locust"]
        
        cmd_parts = ["locust"]
        cmd_parts.append(f"-f {locust_config['locustfile']}")
        cmd_parts.append(f"--host={locust_config['host']}")
        cmd_parts.append("--headless")
        cmd_parts.append(f"-u {locust_config['users']}")
        cmd_parts.append(f"-r {locust_config['spawn_rate']}")
        cmd_parts.append(f"-t {locust_config['duration']}")
        
                       
        if locust_config.get("tags"):
            tags = locust_config["tags"]
            if isinstance(tags, list):
                tags = ",".join(tags)
            cmd_parts.append(f"--tags {tags}")
        
                                                  
        csv_prefix = locust_config.get("csv_prefix", "locust")
        cmd_parts.append(f"--csv={output_dir}/{csv_prefix}")
        
                                                   
        if locust_config.get("html_report"):
            cmd_parts.append(f"--html={output_dir}/{locust_config['html_report']}")
        
        return " ".join(cmd_parts)

    def start_locust(self) -> Tuple[bool, str]:
        """Start Locust on the worker VM."""
        locust_config = self.config["locust"]
        workdir = locust_config["workdir"]
        
                                                                              
        remote_results_dir = f"{workdir}/results/{self.run_id}"
        
        self.logger.info(f"Starting Locust on worker VM...")
        self.logger.info(f"  Remote output dir: {remote_results_dir}")
        
        if not self.args.dry_run:
                                                      
            rc, out, err = self.ssh.run(f"mkdir -p {remote_results_dir}", timeout=30)
            if rc != 0:
                self.logger.error(f"Failed to create remote output dir: {err}")
                return False, ""
        
                                       
        locust_cmd = self.build_locust_command(remote_results_dir)
        
                                                                    
        output_file = f"{remote_results_dir}/locust_output.log"
        
                                     
        env_vars = locust_config.get("env_vars", {})
        env_prefix = " ".join([f"{k}={v}" for k, v in env_vars.items()])
        
                                         
        venv_path = locust_config.get("venv_path", "")
        if venv_path:
            venv_activate = f"source {venv_path}/bin/activate && "
        else:
            venv_activate = ""
        
                                                        
        full_cmd = f"cd {workdir} && {venv_activate}{env_prefix} {locust_cmd}".strip()
        
        self.logger.info(f"  Command: {locust_cmd}")
        self.logger.info(f"  Workdir: {workdir}")
        if venv_path:
            self.logger.info(f"  Venv: {venv_path}")
        self.logger.debug(f"  Full remote command: {full_cmd}")
        
                                         
        self.meta["locust"]["command"] = locust_cmd
        self.meta["locust"]["workdir"] = workdir
        self.meta["locust"]["env_vars"] = env_vars
        self.meta["locust"]["remote_output_dir"] = remote_results_dir
        self.meta["locust"]["remote_output_file"] = output_file
        
        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would start Locust")
            return True, "dry-run"
        
                             
        rc, out, err = self.ssh.run_background(full_cmd, output_file)
        
        if rc == 0 and out.strip():
            pid = out.strip().split("\n")[-1]
            self.locust_pid = pid
            self.meta["locust"]["pid"] = pid
            self.logger.info(f"  Locust started with PID: {pid}")
            return True, pid
        else:
            self.logger.error(f"  Failed to start Locust: {err}")
            self.errors.append(f"Failed to start Locust: {err}")
            return False, ""

    def check_locust_running(self) -> bool:
        """Check if Locust is still running."""
        if not self.locust_pid:
            return False
        
        rc, out, err = self.ssh.run(f"ps -p {self.locust_pid} -o pid= 2>/dev/null")
        return rc == 0 and self.locust_pid in out

    def stop_locust(self):
        """Stop Locust if still running."""
        if not self.locust_pid:
            return
        
        self.logger.info("Stopping Locust...")
        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would stop Locust")
            return
        
                                       
        self.ssh.run(f"kill {self.locust_pid} 2>/dev/null || true", timeout=10)
        time.sleep(2)
        
                                     
        if self.check_locust_running():
            self.ssh.run(f"kill -9 {self.locust_pid} 2>/dev/null || true", timeout=10)

    def inject_anomaly(self) -> bool:
        """Apply the anomaly YAML on the worker VM."""
        anomaly_id = self.config["anomaly"]["anomaly_id"]
        anomaly_info = ANOMALY_CATALOG[anomaly_id]
        anomaly_dir = self.config["anomaly"]["anomaly_dir"]
        namespace = self.config["anomaly"].get("namespace", "tt")
        
        yaml_path = f"{anomaly_dir}/{anomaly_info['filename']}"
        
        self.logger.info(f"Injecting anomaly: {anomaly_id}")
        self.logger.info(f"  Layer: {anomaly_info['layer']}")
        self.logger.info(f"  Type: {anomaly_info['root_cause_type']}")
        self.logger.info(f"  YAML: {yaml_path}")
        
                                                       
        target_service = self.config["anomaly"].get("target_service")
        default_service = anomaly_info.get("default_target_service")
        effective_service = target_service or default_service

                       
        self.meta["anomaly"]["anomaly_id"] = anomaly_id
        self.meta["anomaly"]["filename"] = anomaly_info["filename"]
        self.meta["anomaly"]["yaml_path"] = yaml_path
        self.meta["anomaly"]["layer"] = anomaly_info["layer"]
        self.meta["anomaly"]["root_cause_type"] = anomaly_info["root_cause_type"]
        self.meta["anomaly"]["injection_tool"] = anomaly_info["injection_tool"]
        self.meta["anomaly"]["description"] = anomaly_info["description"]
        self.meta["anomaly"]["namespace"] = namespace
        self.meta["anomaly"]["target_service"] = effective_service
        self.meta["anomaly"]["fault_offset_sec"] = self.config["anomaly"].get("fault_offset_sec")
        self.meta["anomaly"]["fault_duration_sec"] = self.config["anomaly"].get("fault_duration_sec")

                                                                                              
        needs_patch = (
            target_service
            and default_service
            and target_service != default_service
        )
        if needs_patch:
                                                                                        
                                                                                       
            apply_cmd = (
                f"sed 's/{default_service}/{target_service}/g' {yaml_path}"
                f" | kubectl apply -f -"
            )
            self.logger.info(f"  Patching YAML target: {default_service} → {target_service}")
        else:
            apply_cmd = f"kubectl apply -f {yaml_path}"

        self.meta["anomaly"]["apply_command"] = apply_cmd

        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would inject anomaly")
            return True

        rc, out, err = self.ssh.run(apply_cmd, timeout=60)

        if rc == 0:
            self.anomaly_applied = True
            self.logger.info(f"  Anomaly injected successfully (target: {effective_service})")
            self.logger.debug(f"  Output: {out}")
            return True
        else:
            self.logger.error(f"  Failed to inject anomaly: {err}")
            self.errors.append(f"Failed to inject anomaly: {err}")
            return False

    def recover_anomaly(self) -> bool:
        """Delete the anomaly YAML (cleanup)."""
        if not self.anomaly_applied and not self.args.dry_run:
            return True
        
        anomaly_id = self.config["anomaly"]["anomaly_id"]
        anomaly_info = ANOMALY_CATALOG.get(anomaly_id, {})
        anomaly_dir = self.config["anomaly"]["anomaly_dir"]
        
        yaml_path = f"{anomaly_dir}/{anomaly_info.get('filename', '')}"
        
        self.logger.info(f"Recovering from anomaly (kubectl delete)...")
        
        delete_cmd = f"kubectl delete -f {yaml_path} --ignore-not-found=true"
        self.meta["anomaly"]["delete_command"] = delete_cmd
        
        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would recover from anomaly")
            return True
        
        rc, out, err = self.ssh.run(delete_cmd, timeout=60)
        
                                                               
        if rc == 0 or "NotFound" in err or "not found" in err.lower():
            self.anomaly_applied = False
            self.logger.info("  Anomaly recovered successfully")
            return True
        else:
            self.logger.warning(f"  Recovery may have failed: {err}")
                                        
            return True

    def verify_locust_artifacts(self) -> bool:
        """Verify Locust output files exist on worker VM."""
        locust_config = self.config["locust"]
        csv_prefix = locust_config.get("csv_prefix", "locust")
        remote_dir = self.meta["locust"].get("remote_output_dir", "")
        
        self.logger.info("Verifying Locust artifacts on worker VM...")
        self.logger.info(f"  Remote directory: {remote_dir}")
        
        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would verify Locust artifacts")
            return True
        
        if not remote_dir:
            self.logger.warning("  No remote output directory recorded")
            return False
        
        found = []
        missing = []
        
                                                   
        rc, out, err = self.ssh.run(f"ls -la {remote_dir}/ 2>/dev/null || echo 'Directory not found'", timeout=30)
        self.logger.debug(f"Remote Locust files:\n{out}")
        
                         
        for suffix in ["_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv"]:
            remote_path = f"{remote_dir}/{csv_prefix}{suffix}"
            if self.ssh.file_exists(remote_path):
                found.append(f"{csv_prefix}{suffix}")
            else:
                missing.append(f"{csv_prefix}{suffix}")
        
                           
        if locust_config.get("html_report"):
            html_path = f"{remote_dir}/{locust_config['html_report']}"
            if self.ssh.file_exists(html_path):
                found.append(locust_config["html_report"])
            else:
                missing.append(locust_config["html_report"])
        
                                 
        log_path = f"{remote_dir}/locust_output.log"
        if self.ssh.file_exists(log_path):
            found.append("locust_output.log")
        else:
            missing.append("locust_output.log")
        
                                                                        
        self.meta["locust"]["artifacts_found"] = found
        self.meta["locust"]["artifacts_missing"] = missing
        self.meta["locust"]["artifacts_location"] = "VM-TT (worker)"
        
        if found:
            self.logger.info(f"  Found {len(found)} artifacts: {found}")
        if missing:
            self.logger.warning(f"  Missing {len(missing)} artifacts: {missing}")
        
        return len(found) > 0

    def export_telemetry(self, start: datetime, end: datetime):
        """Export metrics, logs, and traces for the experiment window."""
        self.logger.info(f"Exporting telemetry from {utc_iso(start)} to {utc_iso(end)}")
        
        if self.args.dry_run:
            self.logger.info("[DRY-RUN] Would export telemetry")
            return
        
        export_config = self.config["export"]
        
                                   
        prom_config = export_config["prometheus"]
        prom = PrometheusExporter(prom_config["base_url"], self.logger)
        
        prom_results = prom.export_queries(
            queries=prom_config.get("queries", {}),
            start=start,
            end=end,
            step=prom_config.get("step_seconds", 15),
            output_dir=self.run_dir / "metrics"
        )
        self.meta["export"]["prometheus"] = prom_results
        
                                               
        otel_logs_config = export_config["otel_logs"]
        log_fetcher = OTelFileLogFetcher(
            self.ssh,
            otel_logs_config["remote_path"],
            self.logger,
        )
        log_results = log_fetcher.fetch_window(
            start=start,
            end=end,
            output_path=self.run_dir / "logs" / "logs.jsonl",
        )
        self.meta["export"]["otel_logs"] = log_results
        
                              
        jaeger_config = export_config["jaeger"]
        jaeger = JaegerExporter(jaeger_config["base_url"], self.logger)

                                                                                             
        export_delay = jaeger_config.get("export_delay_sec", 0)
        if export_delay > 0:
            self.logger.info(
                f"Waiting {export_delay}s for Jaeger span indexing to complete..."
            )
            self._wait_interruptible(export_delay)

                               
        services = jaeger_config.get("services", [])
        if not services:
                                              
            all_services = jaeger.get_services()
            services = [s for s in all_services if s.startswith("ts-")]

        jaeger_results = jaeger.export_traces(
            services=services,
            start=start,
            end=end,
            limit_per_service=jaeger_config.get("limit_per_service", 50),
            output_dir=self.run_dir / "traces",
            fetch_full_traces=jaeger_config.get("fetch_full_traces", True),
            max_total_traces=jaeger_config.get("max_total_traces", 500),
            request_delay_sec=jaeger_config.get("request_delay_sec", 0.2)
        )
        self.meta["export"]["jaeger"] = jaeger_results

                                                                                 
        if jaeger_results.get("traces_exported", 0) == 0:
            self.logger.warning(
                "WARNING: 0 traces exported from Jaeger. "
                "Possible causes: Jaeger still recovering from fault load, "
                "connection errors, or no traffic during experiment window. "
                "This run's trace data is UNUSABLE for training."
            )
            self.errors.append("Zero traces exported from Jaeger")

    def save_meta(self):
        """Save meta.json to run directory."""
        self.meta["errors"] = self.errors
        meta_path = self.run_dir / "meta.json"
        
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        
        self.logger.info(f"Saved metadata to {meta_path}")

    def cleanup(self):
        """Emergency cleanup handler."""
        if self.cleanup_done:
            return
        self.cleanup_done = True
        
        if self.logger:
            self.logger.warning("Running cleanup...")
        
                         
        try:
            self.recover_anomaly()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cleanup: anomaly recovery failed: {e}")
        
                     
        try:
            self.stop_locust()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cleanup: Locust stop failed: {e}")

    def run(self) -> int:
        """Execute the full experiment."""
                                   
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda s, f: self._signal_handler(s, f))
        signal.signal(signal.SIGTERM, lambda s, f: self._signal_handler(s, f))
        
        try:
                   
            if not self.setup():
                return 1
            
                                   
            if not self.validate_connectivity():
                self.logger.error("Connectivity validation failed. Aborting.")
                self.meta["exit_status"] = "validation_failed"
                self.save_meta()
                return 1
            
                                     
            locust_duration = parse_duration(self.config["locust"]["duration"])
            fault_offset = self.config["anomaly"]["fault_offset_sec"]
            fault_duration = self.config["anomaly"]["fault_duration_sec"]
            pre_buffer = self.config["export"].get("pre_buffer_sec", 60)
            post_buffer = self.config["export"].get("post_buffer_sec", 60)
            
                             
            if fault_offset + fault_duration >= locust_duration:
                self.logger.error(
                    f"Invalid timing: fault_offset({fault_offset}) + fault_duration({fault_duration}) "
                    f">= locust_duration({locust_duration}). Anomaly must end before Locust."
                )
                self.meta["exit_status"] = "invalid_timing"
                self.save_meta()
                return 1
            
            self.logger.info(f"Timing configuration:")
            self.logger.info(f"  Locust duration: {locust_duration}s")
            self.logger.info(f"  Fault offset: {fault_offset}s")
            self.logger.info(f"  Fault duration: {fault_duration}s")
            self.logger.info(f"  Pre-buffer: {pre_buffer}s")
            self.logger.info(f"  Post-buffer: {post_buffer}s")
            
                                                                          
                                   
                                                                          
            locust_start = utc_now()
            self.meta["timestamps"]["locust_start"] = utc_iso(locust_start)
            
            ok, pid = self.start_locust()
            if not ok:
                self.meta["exit_status"] = "locust_start_failed"
                self.save_meta()
                return 1
            
                                                                          
                                              
                                                                          
            self.logger.info(f"Waiting {fault_offset}s before injection...")
            if not self._wait_interruptible(fault_offset):
                self.meta["exit_status"] = "interrupted"
                self.save_meta()
                return 1
            
                                                                          
                                     
                                                                          
            inject_start = utc_now()
            self.meta["timestamps"]["inject_start"] = utc_iso(inject_start)
            
            if not self.inject_anomaly():
                self.logger.warning("Anomaly injection failed, continuing experiment...")
            
                                                                          
                                              
                                                                          
            self.logger.info(f"Fault active for {fault_duration}s...")
            if not self._wait_interruptible(fault_duration):
                self.meta["exit_status"] = "interrupted"
                self.save_meta()
                return 1
            
            inject_end = utc_now()
            self.meta["timestamps"]["inject_end"] = utc_iso(inject_end)
            
                                                                          
                                           
                                                                          
            recover_start = utc_now()
            self.meta["timestamps"]["recover_start"] = utc_iso(recover_start)
            
            self.recover_anomaly()
            
                                                                          
                                                
                                                                          
            elapsed = (utc_now() - locust_start).total_seconds()
            remaining = max(0, locust_duration - elapsed)
            
            self.logger.info(f"Waiting {remaining:.0f}s for Locust to finish...")
            if not self._wait_interruptible(remaining):
                self.meta["exit_status"] = "interrupted"
                self.save_meta()
                return 1
            
                                                    
            time.sleep(5)
            
            locust_end = utc_now()
            self.meta["timestamps"]["locust_end"] = utc_iso(locust_end)
            
                                                                          
                                        
                                                                          
            self.verify_locust_artifacts()
            
                                                                          
                                       
                                                                          
            export_start = locust_start - __import__('datetime').timedelta(seconds=pre_buffer)
            export_end = locust_end + __import__('datetime').timedelta(seconds=post_buffer)
            
            self.meta["timestamps"]["export_window_start"] = utc_iso(export_start)
            self.meta["timestamps"]["export_window_end"] = utc_iso(export_end)
            
            self.export_telemetry(export_start, export_end)
            
                                                                          
                               
                                                                          
                                     
            anomaly_info = ANOMALY_CATALOG.get(self.config["anomaly"]["anomaly_id"], {})
            self.meta["root_cause_layer"] = anomaly_info.get("layer", "unknown")
            self.meta["root_cause_type"] = anomaly_info.get("root_cause_type", "unknown")
            self.meta["affected_services"] = self.config["anomaly"].get("affected_services", [])
            self.meta["injection_tool"] = anomaly_info.get("injection_tool", "unknown")
            
            self.meta["exit_status"] = "success" if not self.errors else "completed_with_errors"
            self.save_meta()
            
            self.logger.info("=" * 60)
            self.logger.info(f"Experiment completed: {self.run_id}")
            self.logger.info(f"Output directory: {self.run_dir}")
            self.logger.info(f"Exit status: {self.meta['exit_status']}")
            self.logger.info("=" * 60)
            
            return 0 if not self.errors else 1
            
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Unhandled exception: {e}")
            self.errors.append(f"Unhandled exception: {e}")
            self.meta["exit_status"] = "exception"
            self.save_meta()
            return 1
        
        finally:
            self.cleanup()

    def _wait_interruptible(self, seconds: float) -> bool:
        """Wait for specified seconds, interruptible by shutdown event."""
        if seconds <= 0:
            return True
        
        if self.args.dry_run:
            self.logger.info(f"[DRY-RUN] Would wait {seconds}s")
            return True
        
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._shutdown_event.is_set():
                return False
            time.sleep(min(1.0, end_time - time.time()))
        return True

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        if self.logger:
            self.logger.warning(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_event.set()


                                                                               
      
                                                                               

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Use JSON config or: pip3 install pyyaml")
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Production-grade experiment runner for multi-modal RCA dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python3 run_experiment.py exp.json
  
  # Dry run (no actual execution)
  python3 run_experiment.py exp.json --dry-run
  
  # Override injection timing
  python3 run_experiment.py exp.json --injection-start-time 180 --anomaly-duration 90
  
  # Verbose output
  python3 run_experiment.py exp.json -v

Available anomaly IDs:
  L1_01_node_cpu_pressure        L2_01_dns_failure
  L1_02_disk_io_saturation       L2_02_network_latency
  L1_03_memory_pressure          L2_03_packet_loss
  L3_01_pod_kill                 L4_01_jvm_memory_pressure
  L3_02_liveness_probe_failure   L4_02_database_connection_exhaustion
  L3_03_resource_quota_exhaustion L4_03_http_error_injection
"""
    )
    
    parser.add_argument(
        "config_file",
        help="Path to experiment config file (JSON or YAML)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would be done without executing"
    )
    
    parser.add_argument(
        "--injection-start-time",
        type=int,
        metavar="SECONDS",
        help="Override fault injection start time (seconds after Locust start)"
    )
    
    parser.add_argument(
        "--anomaly-duration",
        type=int,
        metavar="SECONDS",
        help="Override anomaly duration in seconds"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug output"
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-generated run ID (used by batch_experiment.py "
             "to produce <anomaly>_sample<NN>_<service>_<timestamp> dirs)"
    )
    
    parser.add_argument(
        "--list-anomalies",
        action="store_true",
        help="List all available anomaly IDs and exit"
    )
    
    args = parser.parse_args()
    
                             
    if args.list_anomalies:
        print("\nAvailable Anomaly IDs:\n")
        for aid, info in sorted(ANOMALY_CATALOG.items()):
            print(f"  {aid}")
            print(f"    Layer: {info['layer']}")
            print(f"    Type: {info['root_cause_type']}")
            print(f"    Tool: {info['injection_tool']}")
            print(f"    File: {info['filename']}")
            print()
        return 0
    
                 
    try:
        config = load_config(args.config_file)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1
    
                    
    runner = ExperimentRunner(config, args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
