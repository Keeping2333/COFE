#!/usr/bin/env bash




ANOMALY_ID="L4_01_container_memory_pressure"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"





MEM_LIMIT="${SN_MEM_LIMIT:-16m}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        cname=$(find_container "$TARGET") || die "container $TARGET not found"

        save_container_state "$run_id" "$cname"
        log "squeezing memory of $cname to $MEM_LIMIT for ${duration}s"
        docker update --memory "$MEM_LIMIT" --memory-swap -1 "$cname" >/dev/null 2>&1 \
            || log "docker update failed for $cname"



        echo "$cname" > "$(state_dir "$run_id")/L4_01_squeezed_${TARGET}"
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"mem\":\"$MEM_LIMIT\",\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        for f in "$d"/L4_01_squeezed_*; do
            [ -f "$f" ] || continue
            cname=$(cat "$f")
            log "restoring memory limit of $cname"
            restore_container_state "$run_id" "$cname"
            rm -f "$f"
        done
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 1
        mem=$(docker inspect -f '{{.HostConfig.Memory}}' "$cname")

        [ "$mem" -gt 0 ] && [ "$mem" -le 536870912 ] && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
