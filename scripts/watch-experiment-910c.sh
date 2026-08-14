#!/bin/sh
set -eu

EXPERIMENT_ID=${1:-${AKG_EXPERIMENT_ID:-exp_910c_deepseek_v4_pro_cold_sft_v1}}
PROJECT_ROOT=${AKG_PROJECT_ROOT:-/opt/ascend-kernel-lab}
CONTROLLER_NAME=${AKG_CONTROLLER_CONTAINER:-ascend-kernel-controller}
WORKER_PREFIX=${AKG_WORKER_CONTAINER_PREFIX:-ascend-kernel-worker}
DEVICE_IDS=${AKG_DEVICE_IDS:-0,2,4,6,8,10,12,14}
CONFIG_REQUESTED=${AKG_CONFIG_PATH:-}
DETAIL_LIMIT=${AKG_WATCH_DETAIL_LIMIT:-24}
RUN_ROOT=$PROJECT_ROOT/runs/$EXPERIMENT_ID
START_EPOCH=$(date +%s)

DEFAULT_TASKS='k01_vector_add
k02_bias_gelu
k03_swiglu
k04_transpose
k05_row_softmax
k06_rmsnorm
k07_layernorm
k08_rope
k09_gemm
k10_gemm_bias_gelu'

task_ids() {
    python3 - "$PROJECT_ROOT/runs/metadata.db" "$RUN_ROOT" "$EXPERIMENT_ID" \
        "$CONFIG_REQUESTED" "$PROJECT_ROOT/task_specs/catalog_112.json" \
        $DEFAULT_TASKS <<'PY'
import json
import pathlib
import sqlite3
import sys

database_path = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
experiment_id = sys.argv[3]
config_requested = sys.argv[4]
catalog_path = pathlib.Path(sys.argv[5])
fallback = sys.argv[6:]
selected = []

if database_path.is_file():
    try:
        connection = sqlite3.connect(
            f"file:{database_path}?mode=ro", uri=True, timeout=2.0
        )
        try:
            selected = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT task_id FROM tasks
                    WHERE experiment_id = ? ORDER BY task_id
                    """,
                    (experiment_id,),
                ).fetchall()
            ]
        finally:
            connection.close()
    except sqlite3.Error:
        selected = []

if not selected:
    manifest_path = run_root / "experiment.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        configured = document.get("experiment", {}).get("tasks", [])
        if isinstance(configured, list) and all(
            isinstance(value, str) and value for value in configured
        ):
            selected = configured
    except (OSError, TypeError, json.JSONDecodeError):
        selected = []

if not selected and "deepseek_112" in config_requested and catalog_path.is_file():
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selected = [
            str(task["id"])
            for task in catalog.get("tasks", [])
            if isinstance(task, dict) and isinstance(task.get("id"), str)
        ]
    except (OSError, TypeError, KeyError, json.JSONDecodeError):
        selected = []

for task_id in selected or fallback:
    print(task_id)
PY
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

container_exit_diagnostics() {
    name=$1
    docker container inspect --format \
        'name={{.Name}} running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} restarts={{.RestartCount}} error={{json .State.Error}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} memory={{.HostConfig.Memory}} pids={{.HostConfig.PidsLimit}} log={{.HostConfig.LogConfig.Type}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' \
        "$name" 2>/dev/null || echo "name=$name absent"
}

model_active_count() {
    docker top "$CONTROLLER_NAME" -eo args 2>/dev/null |
        awk 'NR > 1 && /(^|[ /])claude([ ]|$)/ { count += 1 } END { print count + 0 }'
}

task_snapshot() {
    python3 - "$PROJECT_ROOT/runs/metadata.db" "$RUN_ROOT" "$EXPERIMENT_ID" \
        "$DETAIL_LIMIT" "$@" <<'PY'
import json
import pathlib
import re
import sqlite3
import sys

database_path = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
experiment_id = sys.argv[3]
detail_limit = int(sys.argv[4])
task_ids = sys.argv[5:]

stage_rank = {
    "SOURCE_CHECK": 1,
    "FULL_EVALUATION": 4,
    "COMPILE": 2,
    "CORRECTNESS": 3,
    "BENCHMARK": 4,
    "PROFILE": 5,
}
stage_label = {
    "SOURCE_CHECK": "SOURCE",
    "FULL_EVALUATION": "EVAL",
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
        maximum_rounds = optimization_budget
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
                  'SOURCE_CHECK', 'FULL_EVALUATION', 'COMPILE', 'CORRECTNESS',
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


def round_phase(round_root):
    phase_name = None
    phase_index = None
    repair_attempt = None
    prompt_path = round_root / "prompt.json"
    if prompt_path.is_file():
        try:
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            metadata = prompt.get("metadata", {})
            phase_name = metadata.get("phase")
            phase_index = metadata.get("phase_index")
            user_prompt = prompt.get("user_prompt", {})
            if isinstance(user_prompt, dict):
                feedback_state = user_prompt.get("feedback_state", {})
                if isinstance(feedback_state, dict):
                    repair_attempt = feedback_state.get("repair_attempt")
        except (OSError, json.JSONDecodeError):
            pass
    evaluation_path = round_root / "evaluation_result.json"
    if evaluation_path.is_file():
        try:
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            phase_name = evaluation.get("trajectory_phase", phase_name)
            phase_index = evaluation.get("phase_index", phase_index)
            repair_attempt = evaluation.get("repair_attempt", repair_attempt)
        except (OSError, json.JSONDecodeError):
            pass
    return phase_name, phase_index, repair_attempt, prompt_path.is_file()


def phase_marker(phase_name, phase_index, repair_attempt):
    if phase_name == "repair":
        return f"[REP{phase_index}]"
    if phase_name == "optimization":
        return f"[OPT{phase_index}]"
    if phase_name == "optimization_repair":
        if repair_attempt not in (None, 0):
            return f"[OPT{phase_index}-REP{repair_attempt}]"
        return f"[OPTREP{phase_index}]"
    return ""


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


rows = []
counts = {
    "final": 0,
    "round_complete": 0,
    "failed_stage": 0,
    "npu": 0,
    "api_queued": 0,
    "api_inflight": 0,
    "active": 0,
    "unstarted": 0,
}

for task_id in task_ids:
    final_path = run_root / "tasks" / task_id / "final_result.json"
    final_rounds = None
    final_status = None
    termination_reason = None
    completed_repairs = 0
    completed_optimizations = 0
    completed_optimization_repairs = 0
    task_root = run_root / "tasks" / task_id
    round_numbers = []
    if task_root.is_dir():
        for path in task_root.iterdir():
            match = re.fullmatch(r"round_(\d+)", path.name)
            if match is not None and path.is_dir():
                round_numbers.append(int(match.group(1)))
                if round_phase(path)[0] == "optimization_repair":
                    completed_optimization_repairs += 1
    if final_path.is_file():
        try:
            final_value = json.loads(final_path.read_text(encoding="utf-8"))
            final_status = str(final_value.get("status", "-"))
            termination_reason = str(final_value.get("termination_reason", "-"))
            completed_repairs = int(final_value.get("repair_rounds", 0))
            completed_optimizations = int(final_value.get("optimization_rounds", 0))
            final_rounds = max(round_numbers, default=0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            final_rounds = None
    fields = [task_id]
    if final_rounds is not None:
        fields.extend(
            (
                f"FINAL={final_status}",
                f"STOP={termination_reason}",
                f"ROUNDS=REP{completed_repairs}/OPT{completed_optimizations}"
                f"/OPTREP{completed_optimization_repairs}",
            )
        )
    display_rounds = max(maximum_rounds, max(round_numbers, default=0))
    api_queued = False
    api_inflight = False
    npu_active = False
    stage_failed = False
    started = bool(round_numbers)
    for round_number in range(1, display_rounds + 1):
        round_root = (
            run_root / "tasks" / task_id / f"round_{round_number:02d}"
        )
        phase_name, phase_index, repair_attempt, prompt_exists = round_phase(
            round_root
        )
        phase = phase_marker(phase_name, phase_index, repair_attempt)
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
            counts["round_complete"] += 1
        elif (round_root / "feedback.json").is_file():
            stage = "完成"
            counts["round_complete"] += 1
        elif (round_root / "candidate.py").is_file():
            stage = queue_stage(task_id, round_number)
        elif (round_root / "model_response.json").is_file():
            stage = "模型已返回"
        elif prompt_exists:
            if (round_root / "model_request_started.json").is_file():
                stage = "API请求中"
            else:
                stage = "等待API槽"
        else:
            stage = "未开始"
        started = started or prompt_exists or stage != "未开始"
        api_queued = api_queued or stage == "等待API槽"
        api_inflight = api_inflight or stage == "API请求中"
        npu_active = npu_active or "排队" in stage or "执行中" in stage
        stage_failed = stage_failed or "失败" in stage
        fields.append(f"R{round_number:02d}{phase}={stage}")
    row = " ".join(fields)
    if final_rounds is not None:
        counts["final"] += 1
        priority = 0 if final_status and "failed" in final_status else 6
    elif stage_failed:
        counts["failed_stage"] += 1
        priority = 0
    elif npu_active:
        counts["npu"] += 1
        priority = 1
    elif api_inflight:
        counts["api_inflight"] += 1
        priority = 2
    elif api_queued:
        counts["api_queued"] += 1
        priority = 3
    elif started:
        counts["active"] += 1
        priority = 4
    else:
        counts["unstarted"] += 1
        priority = 5
    rows.append((priority, task_id, row))

if detail_limit <= 0 or len(rows) <= detail_limit:
    selected_rows = rows
else:
    selected_rows = sorted(rows)[:detail_limit]

round_target = len(rows) * optimization_budget
round_progress = (
    counts["round_complete"] * 100.0 / round_target
    if round_target
    else 0.0
)
print(
    f"TASKS total={len(rows)} final={counts['final']} "
    f"ROUND_PROGRESS={counts['round_complete']}/{round_target} "
    f"({round_progress:.1f}%) "
    f"failed_stage={counts['failed_stage']} npu={counts['npu']} "
    f"api_inflight={counts['api_inflight']} "
    f"api_queued={counts['api_queued']} active={counts['active']} "
    f"unstarted={counts['unstarted']} "
    f"DETAIL={len(selected_rows)}/{len(rows)}"
)
for _, _, row in selected_rows:
    print(row)
PY
}

final_count() {
    count=0
    for task_id in "$@"; do
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
    python3 - "$RUN_ROOT" "$@" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
passed = 0
for task_id in sys.argv[2:]:
    path = root / "tasks" / task_id / "final_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    status = str(result.get("status", "-"))
    if status.startswith("passed"):
        passed += 1

    task_root = root / "tasks" / task_id
    actual_rounds = 0
    optimization_repairs = 0
    if task_root.is_dir():
        for round_root in task_root.iterdir():
            if re.fullmatch(r"round_(\d+)", round_root.name) is None:
                continue
            if not round_root.is_dir():
                continue
            actual_rounds += 1
            evaluation_path = round_root / "evaluation_result.json"
            if evaluation_path.is_file():
                evaluation = json.loads(
                    evaluation_path.read_text(encoding="utf-8")
                )
                if evaluation.get("trajectory_phase") == "optimization_repair":
                    optimization_repairs += 1

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
        f"OPT_REPAIR={optimization_repairs} ACTUAL={actual_rounds} "
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
    selected_tasks=$(task_ids)
    task_total=$(printf '%s\n' "$selected_tasks" | awk 'NF { count += 1 } END { print count + 0 }')
    workers=$(worker_counts)
    model_active=$(model_active_count)
    finals=$(final_count $selected_tasks)
    tasks=$(task_snapshot $selected_tasks)
    controller_running=$(docker container inspect --format '{{.State.Running}}' \
        "$CONTROLLER_NAME" 2>/dev/null || true)
    snapshot="WORKERS=$workers MODEL_ACTIVE=$model_active FINAL=$finals/$task_total CONTROLLER=${controller_running:-absent}
$tasks"

    if [ "$snapshot" != "$last_snapshot" ]; then
        echo "$(date '+%H:%M:%S') ELAPSED=$(elapsed_time) EXPERIMENT=$EXPERIMENT_ID"
        printf '%s\n' "$snapshot"
        last_snapshot=$snapshot
    fi

    if [ "$controller_running" != true ]; then
        if [ "$task_total" -gt 0 ] && [ "$finals" -eq "$task_total" ]; then
            print_final_summary $selected_tasks
            exit 0
        fi
        echo "Controller 已停止，但只生成了 $finals/$task_total 个最终结果。" >&2
        echo '===== CONTAINER EXIT STATES =====' >&2
        container_exit_diagnostics "$CONTROLLER_NAME" >&2
        old_ifs=$IFS
        IFS=,
        for device in $DEVICE_IDS; do
            container_exit_diagnostics "$WORKER_PREFIX-$device" >&2
        done
        IFS=$old_ifs
        echo '===== FILESYSTEM =====' >&2
        df -h "$PROJECT_ROOT" >&2 || true
        echo '===== CONTROLLER LOG =====' >&2
        docker logs --tail 100 "$CONTROLLER_NAME" >&2 || true
        echo '===== WORKER LOG TAILS =====' >&2
        old_ifs=$IFS
        IFS=,
        for device in $DEVICE_IDS; do
            echo "----- $WORKER_PREFIX-$device -----" >&2
            docker logs --tail 30 "$WORKER_PREFIX-$device" >&2 || true
        done
        IFS=$old_ifs
        exit 1
    fi
    sleep 5
done
