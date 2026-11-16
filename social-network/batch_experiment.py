#!/usr/bin/env python3
"""
Batch runner for the social-network round_sn1 dataset.

Mirrors train-ticket/batch_experiment.py:
  - Same checkpoint format (resumeable)
  - Same per-layer cooldowns
  - Same per-anomaly profile (balanced / query / flow / full)

Differences:
  - SN_W_* env vars (not TT_W_*)
  - SN anomaly IDs (incl. L3_01_container_kill, L4_01_container_memory_pressure, etc.)
  - Five in-distribution service targets selected for SN's call graph
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

                                                                             
                                             
                                                                             
                                                                     
                                                                      
                                                                                  
                                                                             
                                                                                   
                                                    
IN_DIST = [
    "post-storage-service",
    "user-timeline-service",
    "home-timeline-service",
    "compose-post-service",
    "social-graph-service",
]

MONGO_TARGETS = [
    "post-storage-mongodb",
    "user-timeline-mongodb",
    "social-graph-mongodb",
    "media-mongodb",                                                                 
    "user-mongodb",
]

                                                                
EXPERIMENT_PLAN = [
                                                                                       
    ("L1_01_node_cpu_pressure",   None, "balanced"),
    ("L1_02_disk_io_saturation",  None, "flow"),
    ("L1_03_memory_pressure",     None, "balanced"),

                                                     
    ("L2_01_dns_failure",         IN_DIST, "balanced"),
    ("L2_02_network_latency",     IN_DIST, "query"),                                  
    ("L2_03_packet_loss",         IN_DIST, "balanced"),

                                      
    ("L3_01_container_kill",         IN_DIST, "balanced"),
    ("L3_02_health_endpoint_block",  IN_DIST, "balanced"),
    ("L3_03_cgroup_quota_squeeze",   None,    "flow"),                               

                           
    ("L4_01_container_memory_pressure",   IN_DIST, "flow"),                                   
    ("L4_02_mongo_connection_exhaustion", MONGO_TARGETS, "flow"),
    ("L4_03_http_error_injection",        IN_DIST, "balanced"),
]


PROFILE_ENV = {
    "balanced": {"SN_W_DEFAULT": "3", "SN_W_QUERY": "2", "SN_W_FLOW": "1", "SN_W_FULL": "1"},
    "query":    {"SN_W_DEFAULT": "1", "SN_W_QUERY": "5", "SN_W_FLOW": "0", "SN_W_FULL": "1"},
    "flow":     {"SN_W_DEFAULT": "1", "SN_W_QUERY": "1", "SN_W_FLOW": "5", "SN_W_FULL": "1"},
    "full":     {"SN_W_DEFAULT": "1", "SN_W_QUERY": "1", "SN_W_FLOW": "1", "SN_W_FULL": "5"},
}


COOLDOWN_BY_LAYER = {
    "L1": 60,                       
    "L2": 60,                            
    "L3": 90,                                
    "L4": 90,                                      
}


def _layer_of(anomaly_id: str) -> str:
    return anomaly_id.split("_")[0]


                                                                             
            
                                                                             
CHECKPOINT_FILE = Path("batch_experiment_checkpoint.json")


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()).get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


                                                                             
        
                                                                             
def run_single(config_path: Path, anomaly_id: str, service, repeat: int,
               profile: str, sample_num: int, dry_run: bool,
               logger: logging.Logger, output_base_dir: str | None = None) -> bool:
    """Update exp.json in place, invoke run_experiment.py, restore original."""
    original_text = config_path.read_text()
    config = json.loads(original_text)

    config["anomaly"]["anomaly_id"] = anomaly_id
    if service is None:
        config["anomaly"]["affected_services"] = []
        config["anomaly"]["target_service"] = None
    else:
        config["anomaly"]["affected_services"] = [service]
        if _layer_of(anomaly_id) == "L1" or anomaly_id == "L3_03_cgroup_quota_squeeze":
            config["anomaly"]["target_service"] = None
        else:
            config["anomaly"]["target_service"] = service

    config.setdefault("locust", {}).setdefault("env_vars", {}).update(PROFILE_ENV[profile])
    if output_base_dir is not None:
        config["output_base_dir"] = output_base_dir
    config_path.write_text(json.dumps(config, indent=2))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if service is None:
        run_id = f"{anomaly_id}_sample{sample_num:02d}_{ts}"
    else:
        run_id = f"{anomaly_id}_sample{sample_num:02d}_{service}_{ts}"

    cmd = ["python3", "run_experiment.py", str(config_path), "--run-id", run_id]
    if dry_run:
        cmd.append("--dry-run")
    logger.info(f"  CMD: {' '.join(cmd)}")

    success = True
    if not dry_run:
        success = subprocess.run(cmd, cwd=config_path.parent).returncode == 0
    config_path.write_text(original_text)
    return success


def build_plan(args) -> list:
    plan = []
    for anomaly_id, services, profile in EXPERIMENT_PLAN:
        if args.only and anomaly_id != args.only:
            continue
        if services is None:
            total = 5 * args.repeats
            for n in range(1, total + 1):
                plan.append((anomaly_id, None, n, profile, n))
        else:
            sample_num = 0
            for service in services:
                for repeat in range(1, args.repeats + 1):
                    sample_num += 1
                    plan.append((anomaly_id, service, repeat, profile, sample_num))
    return plan


def main():
    p = argparse.ArgumentParser(description="social-network batch experiment runner")
    p.add_argument("--config", default="exp.json")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", default=None)
    p.add_argument("--cooldown", type=int, default=None)
    p.add_argument("--batch-name", default=None,
                   help="Subfolder under exp_runs/batch/; default uses output_base_dir from exp.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    logger = logging.getLogger("batch_experiment")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}"); sys.exit(1)

    if args.batch_name:
        output_base_dir = str(Path("exp_runs") / "batch" / args.batch_name)
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Batch output: {output_base_dir}/")
    else:
        output_base_dir = None

    plan = build_plan(args)
    completed = load_checkpoint() if args.resume else set()
    total = len(plan)
    logger.info(f"Plan size: {total}; repeats={args.repeats}; resume={args.resume}; dry_run={args.dry_run}")

    def disp(s):
        return s if s else "NODE-LEVEL"

    print("\n" + "=" * 100)
    print(f"{'#':>4}  {'Anomaly':<40}  {'Service':<30}  {'Profile':<10}  Sample  Rep")
    print("-" * 100)
    for i, (aid, svc, rep, prof, sn) in enumerate(plan, 1):
        key = f"{aid}:{svc}:{rep}"
        status = "SKIP" if key in completed else "RUN "
        print(f"{i:>4}  [{status}] {aid:<36}  {disp(svc):<30}  {prof:<10}  {sn:>4}    {rep}")
    print()

    if args.dry_run:
        logger.info("[DRY-RUN] no actual execution")

    succ = fail = skip = 0
    for i, (aid, svc, rep, prof, sn) in enumerate(plan, 1):
        key = f"{aid}:{svc}:{rep}"
        layer = _layer_of(aid)
        cd = args.cooldown if args.cooldown is not None else COOLDOWN_BY_LAYER[layer]
        if args.resume and key in completed:
            logger.info(f"[{i}/{total}] SKIP {aid} @ {disp(svc)} (rep {rep})"); skip += 1; continue
        logger.info(f"[{i}/{total}] START aid={aid} svc={disp(svc)} prof={prof} sn={sn:02d} rep={rep}")
        t0 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok = run_single(config_path, aid, svc, rep, prof, sn, args.dry_run, logger, output_base_dir)
        if ok:
            succ += 1; completed.add(key); save_checkpoint(completed)
            logger.info(f"[{i}/{total}] OK    aid={aid} svc={disp(svc)} started={t0}")
        else:
            fail += 1
            logger.warning(f"[{i}/{total}] FAIL  aid={aid} svc={disp(svc)}")
        if i < total and not args.dry_run:
            logger.info(f"  Cooldown {cd}s..."); time.sleep(cd)

    logger.info(f"Done: {succ} ok, {fail} failed, {skip} skipped (checkpoint: {CHECKPOINT_FILE})")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
