#!/usr/bin/env bash



ANOMALY_ID="L2_04_connection_reset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"







apply_reset() {
    local pid="$1" verb="$2"
    sudo nsenter -t "$pid" -n iptables "$verb" INPUT -p tcp -m conntrack --ctstate NEW -j REJECT --reject-with tcp-reset 2>/dev/null || true
}

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        cname=$(find_container "$TARGET") || die "container $TARGET not found"
        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        [ -z "$pid" ] || [ "$pid" = "0" ] && die "could not get pid for $cname"

        d=$(state_dir "$run_id")
        echo "$cname|$pid" > "$d/L2_04_target_${TARGET}"

        log "injecting TCP RST on $cname (pid=$pid) for ${duration}s"
        apply_reset "$pid" -I
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"pid\":$pid,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        for f in "$d"/L2_04_target_*; do
            [ -f "$f" ] || continue
            saved=$(cat "$f"); cname="${saved%|*}"; pid="${saved#*|}"
            new_pid=$(docker inspect -f '{{.State.Pid}}' "$cname" 2>/dev/null)
            [ -n "$new_pid" ] && [ "$new_pid" != "0" ] && pid="$new_pid"
            log "removing TCP RST rule on $cname (pid=$pid)"
            apply_reset "$pid" -D
            rm -f "$f"
        done
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 1
        pid=$(docker inspect -f '{{.State.Pid}}' "$cname")
        sudo nsenter -t "$pid" -n iptables -L INPUT -n 2>/dev/null | grep -q "tcp-reset" && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
