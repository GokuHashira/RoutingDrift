.PHONY: help init init-dev lint format test verify check-imports cpu-smoke pull-results clean

PYTHON ?= python3
SCRATCH ?= .cpu-smoke

help:
	@echo "RoutingDrift -- common tasks"
	@echo ""
	@echo "  make init         Install the package (runtime deps only)"
	@echo "  make init-dev     Install with eval/viz/kernels/dev extras"
	@echo "  make lint         Ruff check (no files modified)"
	@echo "  make format       Ruff auto-fix + format"
	@echo "  make test         Run the test suite"
	@echo "  make check-imports  Verify every intra-project import resolves (no execution)"
	@echo "  make cpu-smoke    Run the full pipeline on a tiny CPU model -- no GPU needed"
	@echo "  make verify       Recompute committed drift CSVs from the raw route dumps"
	@echo "  make pull-results Download the Modal results volume into results_modal/"
	@echo "  make clean        Remove caches and CPU smoke-test output"

init:
	$(PYTHON) -m pip install -e .

init-dev:
	$(PYTHON) -m pip install -e ".[all]"

lint:
	ruff check src tests tools

format:
	ruff check --fix src tests tools
	ruff format src tests tools

test:
	pytest -q

check-imports:
	$(PYTHON) tools/check_imports.py

# The regression gate for refactors: builds a tiny MoE checkpoint and runs the real
# pipeline against it. Catches broken imports, argparse wiring, hook attachment, and
# output paths without touching a GPU.
cpu-smoke:
	$(PYTHON) tools/make_tiny_moe.py --out $(SCRATCH)/tiny-olmoe
	$(PYTHON) -m routingdrift.quantization.run_experiment \
		--model_name $(SCRATCH)/tiny-olmoe \
		--precisions fp16 \
		--target_module mlp.gate \
		--output_dir $(SCRATCH)/run \
		--top_k 2 --max_length 32 --skip_heatmaps
	$(PYTHON) -m routingdrift.quantization.verify_reproducibility --results_dir $(SCRATCH)/run

verify:
	$(PYTHON) -m routingdrift.quantization.verify_reproducibility \
		--results_dir results/olmoe_top2_zaratan

# GPU stages write to a Modal volume, never to this machine. Staged into results_modal/
# rather than results/ for two reasons: the write guard blocks anything landing on the
# committed Zaratan reference, and results/ is re-uploaded to the container on every
# `modal run`, so pulling straight into it would ship experiment output back and forth.
# GPU stages write to a Modal volume, never to this machine. Staged into results_modal/
# rather than results/ for two reasons: the write guard blocks anything landing on the
# committed Zaratan reference, and results/ is re-uploaded to the container on every
# `modal run`, so pulling straight into it would ship experiment output back and forth.
#
# Pulled one directory at a time. A whole-volume pull is not available: a remote path of
# '/' fails with "Is a directory", and the '**' glob form is deprecated and rejected.
# Named remote paths work, so the stage directories are listed explicitly.
STAGE_DIRS ?= smoke_top2 probe olmoe_top8 olmoe_sweep olmoe_replay \
              compiler_real kernels_rerun deepseek_v2_lite qwen36_moe

pull-results:
	rm -rf results_modal.partial && mkdir -p results_modal.partial
	@for d in $(STAGE_DIRS); do \
		if modal volume get routingdrift-results $$d results_modal.partial/$$d >/dev/null 2>&1; then \
			echo "  pulled  $$d"; \
		else \
			echo "  absent  $$d"; \
		fi; \
	done
	@modal volume get routingdrift-results mmlu_prompts.txt results_modal.partial/ >/dev/null 2>&1 \
		&& echo "  pulled  mmlu_prompts.txt" || true
	@if [ -z "$$(ls -A results_modal.partial)" ]; then \
		echo "NOTHING PULLED. Check: modal volume ls routingdrift-results"; \
		rm -rf results_modal.partial; exit 1; \
	fi
	rm -rf results_modal && mv results_modal.partial results_modal
	@echo
	@du -sh results_modal/* 2>/dev/null || true
	@echo
	@echo "Staged in results_modal/. Move what you want tracked into results/ by hand."

clean:
	rm -rf $(SCRATCH)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
