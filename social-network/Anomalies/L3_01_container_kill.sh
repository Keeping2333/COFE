#!/usr/bin/env bash



ANOMALY_ID="L3_01_container_kill"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

KILL_INTERVAL="${SN_KILL_INTERVAL:-30}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        sdir=$(state_dir "$run_id")
        cname=$(find_container "$TARGET") || die "container $TARGET not found"

        log "kill loop on $cname every ${KILL_INTERVAL}s for ${duration}s"
        (
            end=$(( $(date +%s) + duration ))
            while [ "$(date +%s)" -lt "$end" ]; do
                docker kill "$cname" >/dev/null 2>&1 || true
                sleep "$KILL_INTERVAL"

                new=$(find_container "$TARGET" 2>/dev/null)
                [ -n "$new" ] && cname="$new"
            done
        ) >"$sdir/L3_01_killloop.log" 2>&1 &
        register_bg_pid "$run_id" $!
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"interval_sec\":$KILL_INTERVAL,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping kill loop"
        kill_bg_pids "$run_id"



        if [ -n "$TARGET" ]; then
            for c in $(docker ps -a --filter "label=com.docker.compose.service=$TARGET" --filter "status=exited" --format '{{.Names}}'); do
                log "  resurrecting $c"
                docker start "$c" >/dev/null 2>&1 || true
            done
        fi
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)





        cname=$(find_container "$TARGET") || exit 0
        state=$(docker inspect -f '{{.State.Status}}' "$cname" 2>/dev/null)
        [ "$state" != "running" ] && exit 0
        started=$(docker inspect -f '{{.State.StartedAt}}' "$cname" 2>/dev/null)
        now=$(date +%s)
        started_ts=$(date -d "$started" +%s 2>/dev/null || echo 0)
        [ "$started_ts" -gt 0 ] && [ $((now - started_ts)) -lt 30 ] && exit 0
        exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
