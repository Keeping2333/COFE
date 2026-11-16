#!/usr/bin/env bash







ANOMALY_ID="L3_03_cgroup_quota_squeeze"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"





DEFAULT_SERVICES="compose-post-service,user-timeline-service,home-timeline-service"
CPU_LIMIT="${SN_SQUEEZE_CPUS:-0.1}"
MEM_LIMIT="${SN_SQUEEZE_MEM:-32m}"

split_targets() {
    local raw="$1"
    [ -z "$raw" ] && raw="$DEFAULT_SERVICES"
    echo "$raw" | tr ',' ' '
}

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        d=$(state_dir "$run_id")

        for svc in $(split_targets "$TARGET"); do
            cname=$(find_container "$svc") || { log "skip $svc: not found"; continue; }
            save_container_state "$run_id" "$cname"
            log "squeezing $cname → cpus=$CPU_LIMIT mem=$MEM_LIMIT"
            docker update --cpus "$CPU_LIMIT" --memory "$MEM_LIMIT" --memory-swap -1 "$cname" >/dev/null 2>&1 \
                || log "docker update failed for $cname"
            echo "$cname" >> "$d/L3_03_squeezed"
        done
        log_event "$run_id" "start" "{\"targets\":\"$TARGET\",\"cpus\":\"$CPU_LIMIT\",\"mem\":\"$MEM_LIMIT\",\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        if [ -f "$d/L3_03_squeezed" ]; then
            while read -r cname; do
                [ -n "$cname" ] || continue
                log "restoring $cname original limits"
                restore_container_state "$run_id" "$cname"
            done < "$d/L3_03_squeezed"
            rm -f "$d/L3_03_squeezed"
        fi
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        ok=1
        for svc in $(split_targets "$TARGET"); do
            cname=$(find_container "$svc") || continue
            cpus=$(docker inspect -f '{{.HostConfig.NanoCpus}}' "$cname")

            [ "$cpus" -gt 0 ] && [ "$cpus" -le 200000000 ] || ok=0
        done
        [ $ok -eq 1 ] && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <comma_separated_services|empty> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
