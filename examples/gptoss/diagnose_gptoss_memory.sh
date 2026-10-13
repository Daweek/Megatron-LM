#!/bin/bash
# Run from Megatron-LM root, inside the same allocation as the training job.
# Usage: bash examples/gptoss/diagnose_gptoss_memory.sh
set -uo pipefail
LAUNCHER=${LAUNCHER:-examples/gptoss/edo_train_gptoss_120b_h200_8gpu.sh}
[[ -f "$LAUNCHER" ]] || { echo "Missing launcher: $LAUNCHER" >&2; exit 1; }
LOG_DIR=${LOG_DIR:-$PWD/runs/gptoss-memory-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$LOG_DIR"
CGROUP_DIR=""
if [[ -r /proc/self/cgroup ]]; then
    cg_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
    if [[ -n "$cg_path" && -d "/sys/fs/cgroup$cg_path" ]]; then
        CGROUP_DIR="/sys/fs/cgroup$cg_path"
    else
        cg_path=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)
        if [[ -n "$cg_path" && -d "/sys/fs/cgroup/memory$cg_path" ]]; then
            CGROUP_DIR="/sys/fs/cgroup/memory$cg_path"
        fi
    fi
fi
snapshot() {
    date -Is
    free -h
    # Parent cgroups can impose a job-wide limit even if the leaf says max.
    local directory="$CGROUP_DIR"
    while [[ -n "$directory" && "$directory" == /sys/fs/cgroup* ]]; do
        echo "cgroup: $directory"
        for file in memory.current memory.peak memory.max memory.events \
            memory.limit_in_bytes memory.usage_in_bytes memory.max_usage_in_bytes \
            memory.failcnt memory.oom_control memory.stat; do
            if [[ -r "$directory/$file" ]]; then
                echo "$file:"
                cat "$directory/$file"
            fi
        done
        [[ "$directory" == /sys/fs/cgroup ]] && break
        directory=$(dirname "$directory")
    done
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    fi
}
echo "Diagnostics: $LOG_DIR"
cat /proc/self/cgroup > "$LOG_DIR/cgroup.txt" 2>/dev/null || true
snapshot > "$LOG_DIR/before.txt" 2>&1
(
    while true; do
        snapshot
        sleep 10
    done
) > "$LOG_DIR/memory.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
MOCK_DATA=1 OPTIMIZER_OFFLOAD_FRACTION=1.0 bash "$LAUNCHER" 2>&1 | tee "$LOG_DIR/training.log"
status=${PIPESTATUS[0]}
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
snapshot > "$LOG_DIR/after.txt" 2>&1
echo "Training exit status: $status; diagnostics saved in $LOG_DIR"
exit "$status"
