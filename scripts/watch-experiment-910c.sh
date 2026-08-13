#!/bin/sh
set -eu

EXPERIMENT_ID=${1:-${AKG_EXPERIMENT_ID:-exp_910c_deepseek_v4_pro_cold_sft_v1}}
PROJECT_ROOT=${AKG_PROJECT_ROOT:-/opt/ascend-kernel-lab}
CONTROLLER_NAME=${AKG_CONTROLLER_CONTAINER:-ascend-kernel-controller}
WORKER_PREFIX=${AKG_WORKER_CONTAINER_PREFIX:-ascend-kernel-worker}
DEVICE_IDS=${AKG_DEVICE_IDS:-0,2,4,6,8,10,12,14}
RUN_ROOT=$PROJECT_ROOT/runs/$EXPERIMENT_ID
START_EPOCH=$(date +%s)

TASKS='k01_vector_add
k02_bias_gelu
k03_swiglu
k04_transpose
k05_row_softmax
k06_rmsnorm
k07_layernorm
k08_rope
k09_gemm
k10_gemm_bias_gelu'

round_stage() {
    round_root=$1
    if [ -f "$round_root/feedback.json" ] || [ -f "$round_root/evaluation_result.json" ]; then
        printf '%s' '完成'
    elif [ -f "$round_root/candidate.py" ]; then
        printf '%s' 'NPU评测'
    elif [ -f "$round_root/model_response.json" ]; then
        printf '%s' '模型已返回'
    elif [ -f "$round_root/prompt.json" ]; then
        printf '%s' '等待模型'
    else
        printf '%s' '未开始'
    fi
}

worker_counts() {
    running=0
    healthy=0
    total=0
    old_ifs=$IFS
    IFS=,
    for device in $DEVICE_IDS; do
        total=$((total + 1))
        state=$(docker container inspect --format \
            '{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' \
            "$WORKER_PREFIX-$device" 2>/dev/null || true)
        case "$state" in
            true*) running=$((running + 1)) ;;
        esac
        case "$state" in
            'true healthy') healthy=$((healthy + 1)) ;;
        esac
    done
    IFS=$old_ifs
    printf '%s/%s running %s/%s healthy' "$running" "$total" "$healthy" "$total"
}

claude_count() {
    docker top "$CONTROLLER_NAME" -eo comm 2>/dev/null |
        awk '$1 == "claude" { count += 1 } END { print count + 0 }'
}

task_snapshot() {
    for task_id in $TASKS; do
        task_root=$RUN_ROOT/tasks/$task_id
        line=$task_id
        round=1
        while [ "$round" -le 5 ]; do
            round_name=$(printf 'round_%02d' "$round")
            stage=$(round_stage "$task_root/$round_name")
            line="$line R$(printf '%02d' "$round")=$stage"
            round=$((round + 1))
        done
        printf '%s\n' "$line"
    done
}

final_count() {
    count=0
    for task_id in $TASKS; do
        if [ -f "$RUN_ROOT/tasks/$task_id/final_result.json" ]; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

elapsed_time() {
    elapsed=$(( $(date +%s) - START_EPOCH ))
    printf '%02d:%02d:%02d' \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
}

print_final_summary() {
    echo "===== FINAL SUMMARY elapsed=$(elapsed_time) ====="
    python3 - "$RUN_ROOT" $TASKS <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
passed = 0
for task_id in sys.argv[2:]:
    path = root / "tasks" / task_id / "final_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    status = str(result.get("status", "-"))
    if status.startswith("passed"):
        passed += 1

    def show(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    print(
        f"{task_id} status={status} "
        f"best_round={show(result.get('best_round'))} "
        f"hidden_correct={show(result.get('hidden_correctness_passed'))} "
        f"hidden_geo={show(result.get('speedup_geomean'))} "
        f"hidden_min={show(result.get('minimum_speedup'))}"
    )
print(f"TOTAL passed={passed} failed={len(sys.argv[2:]) - passed} tasks={len(sys.argv[2:])}")
PY
}

last_snapshot=''
while :; do
    workers=$(worker_counts)
    claude=$(claude_count)
    finals=$(final_count)
    tasks=$(task_snapshot)
    controller_running=$(docker container inspect --format '{{.State.Running}}' \
        "$CONTROLLER_NAME" 2>/dev/null || true)
    snapshot="WORKERS=$workers CLAUDE=$claude FINAL=$finals/10 CONTROLLER=${controller_running:-absent}
$tasks"

    if [ "$snapshot" != "$last_snapshot" ]; then
        echo "$(date '+%H:%M:%S') ELAPSED=$(elapsed_time) EXPERIMENT=$EXPERIMENT_ID"
        printf '%s\n' "$snapshot"
        last_snapshot=$snapshot
    fi

    if [ "$controller_running" != true ]; then
        if [ "$finals" -eq 10 ]; then
            print_final_summary
            exit 0
        fi
        echo "Controller 已停止，但只生成了 $finals/10 个最终结果。" >&2
        echo '===== CONTROLLER LOG =====' >&2
        docker logs --tail 100 "$CONTROLLER_NAME" >&2 || true
        exit 1
    fi
    sleep 5
done
