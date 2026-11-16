#!/usr/bin/env bash




ANOMALY_ID="L2_01_dns_failure"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

apply_dns_block() {
    local pid="$1" verb="$2"
    sudo nsenter -t "$pid" -n iptables "$verb" OUTPUT -p udp --dport 53 -j DROP 2>/dev/null || true
    sudo nsenter -t "$pid" -n iptables "$verb" OUTPUT -p tcp --dport 53 -j DROP 2>/dev/null || true
}

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        [ -z "$TARGET" ] && die "no target container"

        cname=$(find_container "$TARGET") || die "container $TARGET not found"
        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        [ -z "$pid" ] || [ "$pid" = "0" ] && die "could not get pid for $cname"

        d=$(state_dir "$run_id")
        echo "$cname|$pid" > "$d/L2_01_target_${TARGET}"

        log "blocking DNS on $cname (pid=$pid) for ${duration}s"
        apply_dns_block "$pid" -A
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"pid\":$pid,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        for f in "$d"/L2_01_target_*; do
            [ -f "$f" ] || continue
            saved=$(cat "$f"); cname="${saved%|*}"; pid="${saved#*|}"

            new_pid=$(docker inspect -f '{{.State.Pid}}' "$cname" 2>/dev/null)
            [ -n "$new_pid" ] && [ "$new_pid" != "0" ] && pid="$new_pid"
            log "unblocking DNS on $cname (pid=$pid)"
            apply_dns_block "$pid" -D
            rm -f "$f"
        done
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 1

        if docker exec "$cname" sh -c 'nslookup google.com 2>&1 | grep -qi "timed out\|no servers\|connection refused"' 2>/dev/null; then
            log "verify OK: DNS resolution failing inside $cname"
            exit 0
        fi

        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        sudo nsenter -t "$pid" -n iptables -L OUTPUT -n 2>/dev/null | grep -q "dpt:53" \
            && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
