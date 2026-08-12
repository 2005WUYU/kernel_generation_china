# Repository operating rules

- This repository is developed on ordinary workstations and validated on Huawei Ascend 910C hosts. Never claim NPU validation from the fake backend or CPU-only tests.
- Never install, upgrade, downgrade, or replace `torch`, `torch_npu`, Triton-Ascend, CANN, drivers, or firmware from project automation. A 910C venv must use `--system-site-packages`; fail with diagnostics when the platform stack is missing.
- Never commit model credentials, endpoint tenant secrets, `AKG_HIDDEN_SEED`, generated hidden cases, SQLite databases, run artifacts, profiler raw output, or compiler caches.
- The controller may receive model credentials. The Worker and all candidate stage processes must not. Keep controller and worker environment files separate.
- Candidate code is untrusted. Preserve AST checks, fresh stage processes, environment allowlisting, resource/time/output limits, process-group cleanup, symlink/path checks, device locks, correctness gates, and profiler attribution.
- Correctness failure must short-circuit benchmark and profiler. Profiler failure may not erase committed correctness or benchmark results.
- Hidden evaluation happens only after selecting one public-best candidate. Do not use hidden failures to search historical rounds.
- State transitions are a persistence protocol. Commit artifacts atomically before events that reference them; preserve schema versions and restart idempotency.
- Use `apply_patch` for hand edits. Preserve unrelated user changes. Run `./scripts/check-local.sh` before handoff when the development dependencies are available.
- Shell scripts begin with `set -eu`, do not use destructive cleanup, and must pass `sh -n`; use shellcheck when available.
