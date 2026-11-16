#!/usr/bin/env bash



ANOMALY_ID="L3_02_health_endpoint_block"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        cname=$(find_container "$TARGET") || die "container $TARGET not found"

        log "pausing $cname for ${duration}s (will auto-unpause via background timer)"
        docker pause "$cname" >/dev/null 2>&1 || die "pause failed for $cname"






        (sleep "$duration"; docker unpause "$cname" >/dev/null 2>&1 || true) </dev/null >/dev/null 2>&1 &
        register_bg_pid "$run_id" $!

        echo "$cname" > "$(state_dir "$run_id")/L3_02_paused_${TARGET}"
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        for f in "$d"/L3_02_paused_*; do
            [ -f "$f" ] || continue
            cname=$(cat "$f")
            log "unpausing $cname"
            docker unpause "$cname" >/dev/null 2>&1 || true
            rm -f "$f"
        done
        kill_bg_pids "$run_id"
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 1
        verify_paused "$cname"
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
