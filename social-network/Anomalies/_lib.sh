













set -uo pipefail

SN_CHAOS_ROOT="${SN_CHAOS_ROOT:-/tmp/sn-chaos}"


log() { printf '[%s] [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ANOMALY_ID:-?}" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }


state_dir() {
    local run_id="$1"
    local d="$SN_CHAOS_ROOT/$run_id"
    mkdir -p "$d"
    echo "$d"
}


log_event() {
    local run_id="$1" event="$2" extra_json="${3:-{\}}"
    local d
    d="$(state_dir "$run_id")"
    printf '{"ts":"%s","anomaly_id":"%s","event":"%s","extra":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        "${ANOMALY_ID:-?}" \
        "$event" \
        "$extra_json" \
        >> "$d/events.jsonl"
}







find_container() {
    local logical="$1"
    [ -z "$logical" ] && { echo ""; return 1; }


    local exact
    exact=$(docker ps --filter "name=^${logical}$" --format '{{.Names}}' 2>/dev/null | head -1)
    if [ -n "$exact" ]; then echo "$exact"; return 0; fi


    local by_label
    by_label=$(docker ps --filter "label=com.docker.compose.service=${logical}" --format '{{.Names}}' 2>/dev/null | head -1)
    if [ -n "$by_label" ]; then echo "$by_label"; return 0; fi


    local fuzzy
    fuzzy=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E "(^|[-_])${logical}([-_]|$)" | head -1)
    echo "$fuzzy"
    [ -n "$fuzzy" ]
}



save_container_state() {
    local run_id="$1" container="$2"
    local d; d=$(state_dir "$run_id")
    docker inspect "$container" 2>/dev/null \
        | jq '.[0] | {Image: .Config.Image, HostConfig: {CpuShares: .HostConfig.CpuShares, NanoCpus: .HostConfig.NanoCpus, Memory: .HostConfig.Memory, MemorySwap: .HostConfig.MemorySwap, CpuQuota: .HostConfig.CpuQuota, CpuPeriod: .HostConfig.CpuPeriod}}' \
        > "$d/state_${container}.json"
}







force_recreate_container() {
    local container="$1"
    local svc workdir
    svc=$(docker inspect "$container" --format '{{index .Config.Labels "com.docker.compose.service"}}' 2>/dev/null)
    workdir=$(docker inspect "$container" --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' 2>/dev/null)
    if [ -z "$svc" ] || [ -z "$workdir" ]; then
        log "cannot force-recreate $container: missing compose labels (svc='$svc' workdir='$workdir')"
        return 1
    fi
    log "force-recreating $container (svc=$svc) to clear cgroup mutations"
    ( cd "$workdir" && docker compose up -d --force-recreate --no-deps "$svc" >/dev/null 2>&1 ) \
        || { log "force-recreate failed for $svc"; return 1; }
    return 0
}

restore_container_state() {
    local run_id="$1" container="$2"
    local d; d=$(state_dir "$run_id")
    local f="$d/state_${container}.json"
    [ -f "$f" ] || { log "no saved state for $container in $f, skip restore"; return 0; }

    local mem nanocpus
    mem=$(jq -r '.HostConfig.Memory // 0' "$f")
    nanocpus=$(jq -r '.HostConfig.NanoCpus // 0' "$f")



    if [ "$mem" != "0" ] && [ "$mem" != "null" ]; then
        docker update --memory "$mem" --memory-swap -1 "$container" >/dev/null 2>&1 \
            || log "docker update mem restore failed for $container"
    fi
    if [ "$nanocpus" != "0" ] && [ "$nanocpus" != "null" ]; then
        local cpus
        cpus=$(awk -v n="$nanocpus" 'BEGIN { printf "%.3f", n / 1e9 }')
        docker update --cpus "$cpus" "$container" >/dev/null 2>&1 \
            || log "docker update cpu restore failed for $container"
    fi




    local cur_mem cur_cpu
    cur_mem=$(docker inspect -f '{{.HostConfig.Memory}}' "$container" 2>/dev/null || echo 0)
    cur_cpu=$(docker inspect -f '{{.HostConfig.NanoCpus}}' "$container" 2>/dev/null || echo 0)
    local need_recreate=0
    { [ "$mem" = "0" ] || [ "$mem" = "null" ]; } && [ "${cur_mem:-0}" != "0" ] && need_recreate=1
    { [ "$nanocpus" = "0" ] || [ "$nanocpus" = "null" ]; } && [ "${cur_cpu:-0}" != "0" ] && need_recreate=1
    if [ "$need_recreate" = "1" ]; then
        force_recreate_container "$container" \
            || log "WARNING: $container still has limits (mem=$cur_mem cpu=$cur_cpu) and recreate failed"
    fi
}


register_bg_pid() {
    local run_id="$1" pid="$2"
    local d; d=$(state_dir "$run_id")
    echo "$pid" >> "$d/background.pid"
}


kill_bg_pids() {
    local run_id="$1"
    local d; d=$(state_dir "$run_id")
    local f="$d/background.pid"
    [ -f "$f" ] || return 0
    while read -r pid; do
        [ -n "$pid" ] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            sleep 0.3
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done < "$f"
    : > "$f"
}





verify_running() {
    local container="$1"
    docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null | grep -q '^running$'
}

verify_paused() {
    local container="$1"
    docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null | grep -q '^paused$'
}


cleanup_state_dir() {
    local run_id="$1"
    local d="$SN_CHAOS_ROOT/$run_id"
    if [ -d "$d" ] && [ -z "${SN_KEEP_STATE:-}" ]; then

        find "$d" -mindepth 1 -not -name 'events.jsonl' -delete 2>/dev/null || true
    fi
}
