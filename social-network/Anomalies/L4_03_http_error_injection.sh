#!/usr/bin/env bash



ANOMALY_ID="L4_03_http_error_injection"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"


CONCURRENCY="${SN_ERR_CONCURRENCY:-20}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        sdir=$(state_dir "$run_id")
        cname=$(find_container "$TARGET") || die "container $TARGET not found"


        port=$(docker inspect -f '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' "$cname" 2>/dev/null)
        [ -z "$port" ] && port=8080
        url="http://127.0.0.1:${port}"

        log "spawning $CONCURRENCY curl loops against $url for ${duration}s"
        (
            end=$(( $(date +%s) + duration ))
            for i in $(seq 1 $CONCURRENCY); do
                (
                    while [ "$(date +%s)" -lt "$end" ]; do


                        curl -s -o /dev/null -w '' -m 2 \
                            -X POST "$url/wrk2-api/post/compose" \
                            -d '{"junk":"' -H 'Content-Type: application/json' || true
                        curl -s -o /dev/null -w '' -m 2 "$url/__no_such_path_$RANDOM" || true
                    done
                ) &
            done
            wait
        ) >"$sdir/L4_03_flood.log" 2>&1 &
        register_bg_pid "$run_id" $!
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"url\":\"$url\",\"concurrency\":$CONCURRENCY,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping curl flood"
        kill_bg_pids "$run_id"
        pkill -9 -f 'curl.*wrk2-api' 2>/dev/null || true
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)



        max=0
        for _ in 1 2 3 4; do
            n=$(pgrep -c -f 'curl.*wrk2-api' 2>/dev/null)
            [ -z "$n" ] && n=0
            [ "$n" -gt "$max" ] && max=$n
            sleep 0.3
        done
        [ "$max" -ge 1 ] && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
