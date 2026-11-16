#!/usr/bin/env bash





ANOMALY_ID="L2_02_network_latency"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

LATENCY_MS="${SN_LATENCY_MS:-200}"
JITTER_MS="${SN_JITTER_MS:-50}"
INTERFACE="${SN_NETEM_IF:-eth0}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        sdir=$(state_dir "$run_id")
        cname=$(find_container "$TARGET") || die "container $TARGET not found"
        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        [ -z "$pid" ] || [ "$pid" = "0" ] && die "could not get pid for $cname"

        log "applying ${LATENCY_MS}ms±${JITTER_MS}ms delay on $cname (pid=$pid):$INTERFACE for ${duration}s"

        sudo nsenter -t "$pid" -n tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
        sudo nsenter -t "$pid" -n tc qdisc add dev "$INTERFACE" root netem \
            delay "${LATENCY_MS}ms" "${JITTER_MS}ms" distribution normal \
            >"$sdir/L2_02_tc.log" 2>&1 || die "tc qdisc add failed (see $sdir/L2_02_tc.log)"
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"latency_ms\":$LATENCY_MS,\"jitter_ms\":$JITTER_MS,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping netem latency"
        kill_bg_pids "$run_id"
        cname=$(find_container "$TARGET" 2>/dev/null)
        if [ -n "$cname" ]; then
            pid=$(docker inspect -f '{{.State.Pid}}' "$cname" 2>/dev/null)
            [ -n "$pid" ] && [ "$pid" != "0" ] && \
                sudo nsenter -t "$pid" -n tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
        fi
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 1
        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        sudo nsenter -t "$pid" -n tc qdisc show dev "$INTERFACE" 2>/dev/null | grep -q netem && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
