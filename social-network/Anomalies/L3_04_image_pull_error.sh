#!/usr/bin/env bash









ANOMALY_ID="L3_04_image_pull_error"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; TARGET="${2:-}"; ARG3="${3:-}"; ARG4="${4:-}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        cname=$(find_container "$TARGET") || die "container $TARGET not found"
        orig=$(docker inspect -f '{{.Config.Image}}' "$cname")
        [ -z "$orig" ] && die "could not read image for $cname"


        backup="sn-chaos-backup-${TARGET}:$(date +%s)"
        log "saving image $orig as $backup and removing original tag"
        docker tag "$orig" "$backup" || die "tag failed"

        d=$(state_dir "$run_id")
        echo "$cname|$orig|$backup" > "$d/L3_04_swap_${TARGET}"


        docker rmi "$orig" >/dev/null 2>&1 || log "docker rmi $orig had a warning (ok if image is referenced by other tags)"

        docker kill "$cname" >/dev/null 2>&1 || true
        log_event "$run_id" "start" "{\"target\":\"$TARGET\",\"container\":\"$cname\",\"image\":\"$orig\",\"backup\":\"$backup\",\"duration_sec\":$duration}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        d=$(state_dir "$run_id")
        for f in "$d"/L3_04_swap_*; do
            [ -f "$f" ] || continue
            IFS='|' read -r cname orig backup < "$f"
            log "restoring image $orig from $backup, starting $cname"
            docker tag "$backup" "$orig" || log "retag failed"
            docker rmi "$backup" >/dev/null 2>&1 || true

            docker start "$cname" >/dev/null 2>&1 || \
                docker compose -p socialnetwork up -d 2>/dev/null || \
                docker compose up -d 2>/dev/null || true
            rm -f "$f"
        done
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)
        cname=$(find_container "$TARGET") || exit 0
        status=$(docker inspect -f '{{.State.Status}}' "$cname" 2>/dev/null)

        [ "$status" != "running" ] && exit 0 || exit 1
        ;;

    *) echo "usage: $0 {start|stop|verify} <target_container> <duration_sec|run_id> <run_id>" >&2; exit 2 ;;
esac
