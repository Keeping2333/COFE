#!/usr/bin/env bash


ANOMALY_ID="L1_02_disk_io_saturation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; ARG3="${3:-}"; ARG4="${4:-}"

IOTEST_FILE="${SN_IO_TARGET:-/tmp/sn-chaos-iotest.bin}"

case "$CMD" in
    start)
        duration="${ARG3:-60}"
        run_id="${ARG4:-default}"
        state_dir "$run_id" >/dev/null

        IO_SIZE="${SN_IO_SIZE:-2G}"
        IO_JOBS="${SN_IO_JOBS:-2}"
        log "starting fio randwrite (size=$IO_SIZE jobs=$IO_JOBS duration=${duration}s file=$IOTEST_FILE)"

        nohup fio --name=sn-chaos-randwrite \
            --ioengine=libaio --rw=randwrite --bs=64k \
            --size="$IO_SIZE" --runtime="$duration" --time_based \
            --filename="$IOTEST_FILE" --direct=1 --numjobs="$IO_JOBS" \
            --group_reporting \
            >/tmp/sn-chaos/${run_id}/L1_02_fio.log 2>&1 &
        register_bg_pid "$run_id" $!
        log_event "$run_id" "start" "{\"duration_sec\":$duration,\"size\":\"$IO_SIZE\",\"jobs\":$IO_JOBS}"
        ;;

    stop)
        run_id="${ARG3:-default}"
        log "stopping fio + removing $IOTEST_FILE"
        kill_bg_pids "$run_id"
        pkill -9 fio 2>/dev/null || true
        rm -f "$IOTEST_FILE"
        log_event "$run_id" "stop" "{}"
        cleanup_state_dir "$run_id"
        ;;

    verify)

        if command -v iostat >/dev/null; then
            util=$(iostat -dx 1 2 | awk '/^vd/ {print $NF}' | tail -1)
            [ -z "$util" ] && util=0
            awk -v u="$util" 'BEGIN{exit !(u > 50)}' \
                && { log "verify OK: disk util=$util%"; exit 0; } \
                || { log "verify FAIL: disk util=$util%"; exit 1; }
        else

            pgrep -x fio >/dev/null && exit 0 || exit 1
        fi
        ;;

    *) echo "usage: $0 {start|stop|verify} <ignored> <duration|run_id> <run_id>" >&2; exit 2 ;;
esac
