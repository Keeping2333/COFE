#!/usr/bin/env bash


ANOMALY_ID="L1_03_memory_pressure"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; ARG3="${3:-}"; ARG4="${4:-}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        state_dir "$run_id" >/dev/null





        VM_WORKERS="${SN_VM_WORKERS:-1}"
        VM_BYTES="${SN_VM_BYTES:-6G}"
        log "starting stress-ng vm (workers=$VM_WORKERS bytes=$VM_BYTES duration=${duration}s)"

        nohup stress-ng --vm "$VM_WORKERS" --vm-bytes "$VM_BYTES" --timeout "${duration}s" --vm-keep \
            >/tmp/sn-chaos/${run_id}/L1_03_stress.log 2>&1 &
        register_bg_pid "$run_id" $!
        log_event "$run_id" "start" "{\"duration_sec\":$duration,\"workers\":$VM_WORKERS,\"bytes\":\"$VM_BYTES\"}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping stress-ng vm"
        kill_bg_pids "$run_id"
        pkill -9 stress-ng 2>/dev/null || true
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)

        pgrep -x stress-ng >/dev/null && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <ignored> <duration|run_id> <run_id>" >&2; exit 2 ;;
esac
