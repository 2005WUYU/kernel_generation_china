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
    python3 - "$PROJECT_ROOT/runs/metadata.db" "$RUN_ROOT" "$EXPERIMENT_ID" $TASKS <<'PY'
import json
import pathlib
import sqlite3
import sys

database_path = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
experiment_id = sys.argv[3]
task_ids = sys.argv[4:]

stage_rank = {
    "SOURCE_CHECK": 1,
    "COMPILE": 2,
    "CORRECTNESS": 3,
    "BENCHMARK": 4,
    "PROFILE": 5,
}
stage_label = {
    "SOURCE_CHECK": "SOURCE",
    "COMPILE": "COMPILE",
    "CORRECTNESS": "CORRECTNESS",
    "BENCHMARK": "BENCHMARK",
    "PROFILE": "PROFILE",
}
latest = {}
optimization_budget = 5
repair_budget = 0
maximum_rounds = optimization_budget
manifest_path = run_root / "experiment.json"
if manifest_path.is_file():
    try:
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = manifest_document.get("experiment", {})
        optimization_budget = int(manifest.get("rounds_per_task", 5))
        repair_budget = int(manifest.get("maximum_repair_rounds", 0))
        maximum_rounds = optimization_budget + repair_budget
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        optimization_budget = 5
        repair_budget = 0
        maximum_rounds = 5

if database_path.is_file():
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro", uri=True, timeout=2.0
    )
    try:
        rows = connection.execute(
            """
            SELECT task_id, round_number, stage, status,
                   last_error_json, result_json
            FROM evaluation_jobs
            WHERE experiment_id = ?
              AND stage IN (
                  'SOURCE_CHECK', 'COMPILE', 'CORRECTNESS',
                  'BENCHMARK', 'PROFILE'
              )
            """,
            (experiment_id,),
        )
        for row in rows:
            key = (str(row[0]), int(row[1]))
            stage = str(row[2])
            if stage_rank[stage] >= stage_rank.get(latest.get(key, (None,))[0], 0):
                latest[key] = (
                    stage,
                    str(row[3]),
                    row[4],
                    row[5],
                )
    finally:
        connection.close()


def json_object(raw):
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def queue_stage(task_id, round_number):
    value = latest.get((task_id, round_number))
    if value is None:
        return "NPU排队"
    stage, status, raw_error, raw_result = value
    label = stage_label[stage]
    error = json_object(raw_error)
    result = json_object(raw_result)
    error_type = str(error.get("type", ""))
    result_status = str(result.get("status", "")).lower()
    result_error = result.get("error")
    if isinstance(result_error, dict) and not error_type:
        error_type = str(result_error.get("type", ""))

    if error_type == "StageTimeout" or result_status == "timeout":
        return f"{label}超时"
    if status == "QUEUED":
        return f"{label}排队"
    if status == "LEASED":
        return f"{label}执行中"
    if status == "RETRY_WAIT":
        return f"{label}重试"
    if status == "DEAD":
        return f"{label}失败"
    if status == "CANCELLED":
        return f"{label}取消"
    if status == "SUCCEEDED":
        if result_status in {"error", "unavailable", "fail", "failed"} or result.get(
            "passed"
        ) is False:
            return f"{label}失败"
        return f"{label}完成"
    return f"{label}:{status}"


for task_id in task_ids:
    final_path = run_root / "tasks" / task_id / "final_result.json"
    final_rounds = None
    final_status = None
    termination_reason = None
    completed_repairs = 0
    completed_optimizations = 0
    if final_path.is_file():
        try:
            final_value = json.loads(final_path.read_text(encoding="utf-8"))
            final_status = str(final_value.get("status", "-"))
            termination_reason = str(final_value.get("termination_reason", "-"))
            completed_repairs = int(final_value.get("repair_rounds", 0))
            completed_optimizations = int(final_value.get("optimization_rounds", 0))
            final_rounds = completed_repairs + completed_optimizations
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            final_rounds = None
    fields = [task_id]
    if final_rounds is not None:
        fields.extend(
            (
                f"FINAL={final_status}",
                f"STOP={termination_reason}",
                f"ROUNDS=REP{completed_repairs}/OPT{completed_optimizations}",
            )
        )
    for round_number in range(1, maximum_rounds + 1):
        round_root = (
            run_root / "tasks" / task_id / f"round_{round_number:02d}"
        )
        phase = ""
        prompt_path = round_root / "prompt.json"
        if prompt_path.is_file():
            try:
                prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
                metadata = prompt.get("metadata", {})
                phase_name = metadata.get("phase")
                phase_index = metadata.get("phase_index")
                if phase_name in {"repair", "optimization"}:
                    short = "REP" if phase_name == "repair" else "OPT"
                    phase = f"[{short}{phase_index}]"
            except (OSError, json.JSONDecodeError):
                pass
        evaluation_path = round_root / "evaluation_result.json"
        if final_rounds is not None and round_number > final_rounds:
            if final_status == "repair_exhausted":
                stage = "未运行(Repair耗尽/无正确Seed)"
            elif termination_reason == "host_dispatch_limited":
                stage = "未运行(Host瓶颈早停)"
            elif (
                completed_repairs < repair_budget
                and completed_optimizations >= optimization_budget
            ):
                stage = "空槽(Repair提前通过)"
            else:
                stage = "未运行(任务已结束)"
        elif evaluation_path.is_file():
            try:
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                overall_status = str(evaluation.get("overall_status", "unknown"))
            except (OSError, json.JSONDecodeError):
                overall_status = "unknown"
            stage = f"完成[{overall_status}]"
        elif (round_root / "feedback.json").is_file():
            stage = "完成"
        elif (round_root / "candidate.py").is_file():
            stage = queue_stage(task_id, round_number)
        elif (round_root / "model_response.json").is_file():
            stage = "模型已返回"
        elif prompt_path.is_file():
            stage = "等待模型"
        else:
            stage = "未开始"
        fields.append(f"R{round_number:02d}{phase}={stage}")
    print(" ".join(fields))
PY
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
        f"termination={show(result.get('termination_reason'))} "
        f"REP={show(result.get('repair_rounds'))} "
        f"OPT={show(result.get('optimization_rounds'))} "
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
