#!/usr/bin/env bash






ANOMALY_ID="L4_04_mongo_slow_query"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

CHAOS_IMAGE="${SN_MONGO_IMAGE:-mongo:4.4}"
PARALLEL="${SN_SLOW_PARALLEL:-8}"
SIDECAR_NAME="sn-chaos-mongo-slow"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        cname=$(find_container "$TARGET") || die "container $TARGET not found"

        net=$(docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' "$cname" | head -1)
        [ -z "$net" ] && die "could not determine docker network for $cname"

        log "launching $PARALLEL parallel slow ops against $TARGET for ${duration}s"
        docker run -d --rm \
            --name "$SIDECAR_NAME" \
            --network "$net" \
            -e TARGET_HOST="$TARGET" \
            -e PARALLEL="$PARALLEL" \
            -e DURATION="$duration" \
            "$CHAOS_IMAGE" \
            sh -c '
                end=$(( $(date +%s) + DURATION ))
                while [ $(date +%s) -lt $end ]; do
                    for i in $(seq 1 $PARALLEL); do


                        mongo --quiet --host "$TARGET_HOST" --eval "
                            db.adminCommand({ sleep: 1, lock: \"r\", secs: 1 });
                        " >/dev/null 2>&1 &
                    done
                    sleep 1
                done
                wait
            ' \
            >/dev/null 2>&1 || log "sidecar launch failed"

        echo "$SIDECAR_NAME" > "$(state_dir "$run_id")/L4_04_sidecar"
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"net\":\"$net\",\"parallel\":$PARALLEL,\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        if [ -f "$d/L4_04_sidecar" ]; then
            sidecar=$(cat "$d/L4_04_sidecar")
            log "killing sidecar $sidecar"
            docker rm -f "$sidecar" >/dev/null 2>&1 || true
            rm -f "$d/L4_04_sidecar"
        fi
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        docker ps --filter "name=$SIDECAR_NAME" --format '{{.Names}}' | grep -q "$SIDECAR_NAME" && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <mongo_service> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
