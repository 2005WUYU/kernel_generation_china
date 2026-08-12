PYTHON ?= python3
DIST_DIR ?= dist

.PHONY: install test lint typecheck check doctor package package-check

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -m 'not npu'

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

check: lint typecheck test

doctor:
	$(PYTHON) -m ascend_kernel_lab doctor --config configs/experiment_910c_kimi_k3.yaml

package:
	$(PYTHON) -m build --sdist --wheel --outdir "$(DIST_DIR)"

package-check: package
	AKG_DEV_PYTHON='$(PYTHON)' ./scripts/check-package.sh "$(DIST_DIR)"
