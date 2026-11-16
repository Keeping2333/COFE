#!/usr/bin/env python3
"""
Batch Experiment Runner for RCA Dataset Collection
===================================================

Iterates over the EXPERIMENT_PLAN (anomaly_type × target_service combinations)
and calls run_experiment.py for each combination.  A cooldown pause between runs
lets the cluster fully recover before the next injection.

Service targets are chosen from the Train-Ticket architecture diagram:
  - L1 (node-level): high-traffic services that clearly show symptoms
  - L2 (network):    services with many inter-service dependencies
  - L3 (platform):   services that many others depend on
  - L4 (application):Java services with JVM / DB / HTTP characteristics

Usage:
  python3 batch_experiment.py                           # full plan, 1 repeat each
  python3 batch_experiment.py --repeats 3               # 3 runs per (anomaly, service)
  python3 batch_experiment.py --dry-run                 # print plan, no execution
  python3 batch_experiment.py --resume                  # skip completed combos
  python3 batch_experiment.py --only L2_02_network_latency  # single anomaly type
  python3 batch_experiment.py --cooldown 180            # override cooldown (seconds)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

                                                                             
                                                  
 
                                                         
                               
                                                                         
                                                                         
                                                                        
                                                                    
                                                                
                                                                    
                                                                       
                                                          
                                                                      
                     
                                                          
                                           
                                              
                               
                                                    
                                                  
                                                                     
                                                    
                                          
                                               
                                                                             
 
                                                            
                      
                                                                                      
                                                                                       
                                                                                    
                                                                         
                                          
                                                
                                                                    
                                                       
EXPERIMENT_PLAN = [
                                                                        
                                                                                
                                                                                  
                                                                                         
                                                                        
    ("L1_01_node_cpu_pressure",  None, "balanced"),
    ("L1_02_disk_io_saturation", None, "flow"),                                  
    ("L1_03_memory_pressure",    None, "balanced"),

                                                                        
                                                                               
                                                                                 
                                                                        
    ("L2_01_dns_failure", [
        "ts-order-service",                                
        "ts-travel-service",                                           
        "ts-seat-service",
        "ts-route-service",
        "ts-preserve-service",
    ], "balanced"),
    ("L2_02_network_latency", [
        "ts-travel-service",               
        "ts-order-service",
        "ts-seat-service",
        "ts-route-service",
        "ts-station-service",
    ], "query"),                                                
    ("L2_03_packet_loss", [
        "ts-order-service",                                
        "ts-travel-service",
        "ts-seat-service",
        "ts-payment-service",
        "ts-train-service",
    ], "balanced"),

                                                                        
                                                     
                                                                           
                                                                        
    ("L3_01_pod_kill", [                                
        "ts-basic-service",                                               
        "ts-travel-service",
        "ts-order-service",
        "ts-route-service",
        "ts-seat-service",
    ], "balanced"),
    ("L3_02_liveness_probe_failure", [
        "ts-order-service",                
        "ts-travel-service",
        "ts-basic-service",
        "ts-seat-service",
        "ts-route-service",
    ], "balanced"),
    ("L3_03_resource_quota_exhaustion", None, "flow"),
                                                                              
                                                                             

                                                                        
                                               
                                                                      
                                                                        
    ("L4_01_jvm_memory_pressure", [               
        "ts-order-service",                
        "ts-travel-service",
        "ts-seat-service",
        "ts-route-service",
        "ts-train-service",
    ], "full"),                                                     
    ("L4_02_database_connection_exhaustion", [
        "ts-order-service",                
        "ts-travel-service",
        "ts-user-service",
        "ts-seat-service",
        "ts-train-service",
    ], "flow"),                                              
    ("L4_03_http_error_injection", [
        "ts-travel-service",               
        "ts-order-service",
        "ts-seat-service",
        "ts-route-service",
        "ts-payment-service",
    ], "balanced"),
]

                                           
PROFILE_ENV = {
    "balanced": {"TT_W_DEFAULT": "3", "TT_W_QUERY": "2", "TT_W_FLOW": "1", "TT_W_ADMIN": "1", "TT_W_FULL": "1"},
    "query":    {"TT_W_DEFAULT": "1", "TT_W_QUERY": "5", "TT_W_FLOW": "0", "TT_W_ADMIN": "0", "TT_W_FULL": "1"},
    "flow":     {"TT_W_DEFAULT": "1", "TT_W_QUERY": "1", "TT_W_FLOW": "5", "TT_W_ADMIN": "0", "TT_W_FULL": "1"},
    "full":     {"TT_W_DEFAULT": "1", "TT_W_QUERY": "1", "TT_W_FLOW": "1", "TT_W_ADMIN": "1", "TT_W_FULL": "5"},
}

                                                              
COOLDOWN_BY_LAYER = {
    "L1": 180,                                           
    "L2": 120,                            
    "L3": 180,                                 
    "L4": 180,                        
}

                        
def _layer_of(anomaly_id: str) -> str:
    return anomaly_id.split("_")[0]


                                                                             
                    
                                                                             
CHECKPOINT_FILE = Path("batch_experiment_checkpoint.json")


def load_checkpoint() -> set:
    """Return set of completed run keys: 'anomaly_id:service:repeat'."""
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2)
    )


                                                                             
             
                                                                             
def run_single(
    config_path: Path,
    anomaly_id: str,
    service,                                                                    
    repeat: int,
    profile: str,
    sample_num: int,
    dry_run: bool,
    logger: logging.Logger,
    output_base_dir: str = None,
) -> bool:
    """
    Update exp.json in-place, invoke run_experiment.py, return success flag.
    The original exp.json is restored after each run.

    service=None means node-level (L1) or namespace-level (L3_03) fault:
      target_service stays null, affected_services becomes [].
    """
    original_text = config_path.read_text()
    config = json.loads(original_text)

                                            
    config["anomaly"]["anomaly_id"] = anomaly_id
    if service is None:
                                                                              
        config["anomaly"]["affected_services"] = []
        config["anomaly"]["target_service"] = None
    else:
        config["anomaly"]["affected_services"] = [service]
                                                                                             
        layer = _layer_of(anomaly_id)
        if layer == "L1" or anomaly_id == "L3_03_resource_quota_exhaustion":
            config["anomaly"]["target_service"] = None
        else:
            config["anomaly"]["target_service"] = service

                                                                         
    config.setdefault("locust", {}).setdefault("env_vars", {}).update(PROFILE_ENV[profile])

                                                             
    if output_base_dir is not None:
        config["output_base_dir"] = output_base_dir

    config_path.write_text(json.dumps(config, indent=2))

                    
                                                                  
                                                                                            
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if service is None:
        run_id = f"{anomaly_id}_sample{sample_num:02d}_{timestamp}"
    else:
        run_id = f"{anomaly_id}_sample{sample_num:02d}_{service}_{timestamp}"

    cmd = ["python3", "run_experiment.py", str(config_path), "--run-id", run_id]
    if dry_run:
        cmd.append("--dry-run")

    logger.info(
        f"  CMD: {' '.join(cmd)}"
    )

    success = True
    if not dry_run:
        result = subprocess.run(cmd, cwd=config_path.parent)
        success = result.returncode == 0

                             
    config_path.write_text(original_text)
    return success


def build_plan(args) -> list:
    """Return list of (anomaly_id, service_or_None, repeat, profile, sample_num) tuples.

    For per-service anomalies (services is a list):
        sample_num counts 1..(len(services) * repeats); repeat is 1..repeats within service.
    For node-level / namespace-level anomalies (services is None):
        service is None; sample_num and repeat both run 1..(5 * repeats) so class size
        stays equal to per-service anomalies.
    """
    plan = []
    for anomaly_id, services, profile in EXPERIMENT_PLAN:
        if args.only and anomaly_id != args.only:
            continue
        if services is None:
                                                                                
            total = 5 * args.repeats
            for sample_num in range(1, total + 1):
                plan.append((anomaly_id, None, sample_num, profile, sample_num))
        else:
            sample_num = 0
            for service in services:
                for repeat in range(1, args.repeats + 1):
                    sample_num += 1
                    plan.append((anomaly_id, service, repeat, profile, sample_num))
    return plan


def main():
    parser = argparse.ArgumentParser(description="Batch RCA experiment runner")
    parser.add_argument(
        "--config", default="exp.json",
        help="Path to exp.json (default: exp.json)"
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of times to repeat each (anomaly, service) combination (default: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and pass --dry-run to run_experiment.py; no real injection"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip combinations already recorded in the checkpoint file"
    )
    parser.add_argument(
        "--only", default=None,
        help="Run only a specific anomaly_id (e.g. L2_02_network_latency)"
    )
    parser.add_argument(
        "--cooldown", type=int, default=None,
        help="Override cooldown seconds between runs (ignores per-layer defaults)"
    )
    parser.add_argument(
        "--batch-name", default=None,
        help="Name for this batch run (e.g. 'batch1'). Output saved to exp_runs/batch/<name>/. "
             "If omitted, output goes to exp_runs/ (same as single runs)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logger = logging.getLogger("batch_experiment")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

                                             
    if args.batch_name:
        output_base_dir = str(Path("exp_runs") / "batch" / args.batch_name)
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Batch output directory: {output_base_dir}/")
    else:
        output_base_dir = None                               

    plan = build_plan(args)
    completed = load_checkpoint() if args.resume else set()

    total = len(plan)
    logger.info(f"Experiment plan: {total} runs")
    logger.info(f"  Repeats per combo: {args.repeats}")
    logger.info(f"  Resume mode: {args.resume}")
    logger.info(f"  Dry run: {args.dry_run}")

                                                                                   
    def _svc_disp(s):
        return s if s else "NODE-LEVEL"

                         
    print("\n" + "=" * 100)
    print(f"{'#':>4}  {'Anomaly':<40}  {'Service':<30}  {'Profile':<10}  Sample  Rep")
    print("-" * 100)
    for i, (anomaly_id, service, repeat, profile, sample_num) in enumerate(plan, 1):
        key = f"{anomaly_id}:{service}:{repeat}"
        status = "SKIP" if key in completed else "RUN "
        print(f"{i:>4}  [{status}] {anomaly_id:<36}  {_svc_disp(service):<30}  {profile:<10}  {sample_num:>4}    {repeat}")
    print()

    if args.dry_run:
        logger.info("[DRY-RUN] Printing plan only, no actual execution")

             
    succeeded = 0
    failed = 0
    skipped = 0

    for i, (anomaly_id, service, repeat, profile, sample_num) in enumerate(plan, 1):
        key = f"{anomaly_id}:{service}:{repeat}"
        layer = _layer_of(anomaly_id)
        cooldown = args.cooldown if args.cooldown is not None else COOLDOWN_BY_LAYER[layer]

        if args.resume and key in completed:
            logger.info(f"[{i}/{total}] SKIP {anomaly_id} @ {_svc_disp(service)} (rep {repeat})")
            skipped += 1
            continue

                                                                                         
        rep_total = (5 * args.repeats) if service is None else args.repeats
        logger.info(
            f"[{i}/{total}] START  anomaly={anomaly_id}  service={_svc_disp(service)}  "
            f"profile={profile}  sample={sample_num:02d}  repeat={repeat}/{rep_total}"
        )
        start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        ok = run_single(config_path, anomaly_id, service, repeat, profile, sample_num, args.dry_run, logger, output_base_dir)

        if ok:
            succeeded += 1
            completed.add(key)
            save_checkpoint(completed)
            logger.info(
                f"[{i}/{total}] OK     anomaly={anomaly_id}  service={_svc_disp(service)}  "
                f"started={start_ts}"
            )
        else:
            failed += 1
            logger.warning(
                f"[{i}/{total}] FAIL   anomaly={anomaly_id}  service={_svc_disp(service)}  "
                f"(run_experiment.py returned non-zero)"
            )

                                            
        if i < total and not args.dry_run:
            logger.info(f"  Cooling down {cooldown}s before next run...")
            time.sleep(cooldown)

             
    print()
    logger.info("=" * 60)
    logger.info(f"Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    logger.info(f"Checkpoint saved to: {CHECKPOINT_FILE}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
