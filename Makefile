.PHONY: help init init-dev lint format test verify check-imports cpu-smoke clean

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

clean:
	rm -rf $(SCRATCH)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
