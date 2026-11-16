#!/usr/bin/env bash








ANOMALY_ID="L1_01_node_cpu_pressure"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        state_dir "$run_id" >/dev/null

        CPU_LOAD="${SN_CPU_LOAD:-95}"
        CPU_WORKERS="${SN_CPU_WORKERS:-0}"
        log "starting stress-ng cpu (workers=$CPU_WORKERS load=$CPU_LOAD duration=${duration}s)"

        nohup stress-ng --cpu "$CPU_WORKERS" --cpu-load "$CPU_LOAD" --timeout "${duration}s" \
            >/tmp/sn-chaos/${run_id}/L1_01_stress.log 2>&1 &
        register_bg_pid "$run_id" $!
        log_event "$run_id" "start" "{\"duration_sec\":$duration,\"cpu_load\":$CPU_LOAD,\"workers\":$CPU_WORKERS}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping stress-ng cpu"
        kill_bg_pids "$run_id"
        pkill -9 stress-ng 2>/dev/null || true
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)



        read _ u n s i _ _ _ _ _ _ < /proc/stat
        total1=$((u+n+s+i)); idle1=$i
        sleep 1
        read _ u n s i _ _ _ _ _ _ < /proc/stat
        total2=$((u+n+s+i)); idle2=$i
        diff_total=$((total2-total1)); diff_idle=$((idle2-idle1))

        busy_pct=$(awk -v d=$diff_total -v i=$diff_idle 'BEGIN{ printf "%.0f", (1 - i/d) * 100 }')
        threshold="${SN_VERIFY_BUSY_MIN:-50}"
        if [ "$busy_pct" -ge "$threshold" ]; then
            log "verify OK: busy=${busy_pct}% >= ${threshold}%"; exit 0
        else
            log "verify FAIL: busy=${busy_pct}% < ${threshold}%"; exit 1
        fi
        ;;

    *)
        echo "usage: $0 {start|stop|verify} <ignored> <duration|run_id> <run_id>" >&2
        exit 2 ;;
esac
