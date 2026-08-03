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
	@echo "  make pull-results Mirror the Modal results volume into modal_outputs/"
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

# GPU stages write to a Modal volume, never to this machine. Everything on that volume is
# mirrored into modal_outputs/: every run's outputs and every log, minus model weights,
# which live on a separate HF cache volume and never come down. It is gitignored on purpose.
# It is the raw record of what was run; the curated subset is copied into results/ by hand.
#
# Not pulled into results/ directly, for two reasons: the write guard blocks anything
# landing on the committed Zaratan reference, and results/ is re-uploaded to the container
# on every `modal run`, so pulling into it would ship experiment output back and forth.
#
# Pulled one directory at a time. A whole-volume pull is not available: a remote path of
# '/' fails with "Is a directory", and the '**' glob form is deprecated and rejected.
MODAL_OUT ?= modal_outputs
STAGE_DIRS ?= smoke_top2 probe olmoe_top8 olmoe_sweep olmoe_replay \
              compiler_real kernels_rerun deepseek_v2_lite qwen3_30b_a3b

pull-results:
	@# Fail loudly if the CLI is missing or logged out. Without this the per-directory
	@# loop below reports every stage as "absent", because "command not found" matches
	@# the same *"not found"* pattern a genuinely missing directory does -- so a shell
	@# without modal on PATH looks identical to an empty volume.
	@command -v modal >/dev/null 2>&1 || { \
		echo "modal CLI not on PATH. Activate the venv that has it, then retry."; \
		exit 1; }
	@modal volume ls routingdrift-results >/dev/null 2>&1 || { \
		echo "modal cannot read routingdrift-results. Check 'modal token new' and that"; \
		echo "the active workspace is the one that owns the volume:"; \
		modal volume ls routingdrift-results 2>&1 | tail -5; \
		exit 1; }
	rm -rf $(MODAL_OUT).partial && mkdir -p $(MODAL_OUT).partial
	@# Destination is the PARENT directory, not the full target path. modal places the
	@# entry inside it, the same way it handled mmlu_prompts.txt. Passing the full path
	@# creates an empty directory and fails.
	@for d in $(STAGE_DIRS); do \
		out=$$(modal volume get routingdrift-results $$d $(MODAL_OUT).partial/ 2>&1); \
		if [ -d "$(MODAL_OUT).partial/$$d" ] && [ -n "$$(ls -A $(MODAL_OUT).partial/$$d 2>/dev/null)" ]; then \
			echo "  pulled  $$d"; \
		else \
			rm -rf "$(MODAL_OUT).partial/$$d"; \
			case "$$out" in \
				*"command not found"*|*"No module named"*) \
					echo "  ERROR   $$d: modal CLI broken: $$(echo "$$out" | tail -1)" ;; \
				*"not found"*|*"No such"*) echo "  absent  $$d  (stage not run yet)" ;; \
				*) echo "  FAILED  $$d: $$(echo "$$out" | tail -1)" ;; \
			esac; \
		fi; \
	done
	@out=$$(modal volume get routingdrift-results mmlu_prompts.txt $(MODAL_OUT).partial/ 2>&1); \
		[ -f $(MODAL_OUT).partial/mmlu_prompts.txt ] && echo "  pulled  mmlu_prompts.txt" \
		|| echo "  absent  mmlu_prompts.txt"
	@if [ -z "$$(ls -A $(MODAL_OUT).partial)" ]; then \
		echo "NOTHING PULLED. Check: modal volume ls routingdrift-results"; \
		rm -rf $(MODAL_OUT).partial; exit 1; \
	fi
	rm -rf $(MODAL_OUT) && mv $(MODAL_OUT).partial $(MODAL_OUT)
	@# STAGE_DIRS is hardcoded, so a stage writing a new directory would be pulled as
	@# nothing and reported as nothing. Name whatever the volume holds that we did not ask
	@# for, rather than letting it go missing quietly.
	@echo
	@modal volume ls routingdrift-results 2>/dev/null \
	  | sed -E 's/^[^a-zA-Z0-9_.-]*//; s/[[:space:]].*$$//' \
	  | grep -E '^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$$' \
	  | while read -r entry; do \
	      case " $(STAGE_DIRS) mmlu_prompts.txt " in \
	        *" $$entry "*) ;; \
	        *) echo "  NOT MIRRORED: $$entry  (add it to STAGE_DIRS)" ;; \
	      esac; \
	    done
	@echo
	@du -sh $(MODAL_OUT)/* 2>/dev/null || true
	@echo
	@echo "Mirrored into $(MODAL_OUT)/ (gitignored). Copy what should be tracked into results/."

clean:
	rm -rf $(SCRATCH)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
