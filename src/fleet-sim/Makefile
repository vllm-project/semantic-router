# vllm-sr-sim Makefile
#
# Common targets
#   make test         – run the full test suite
#   make test-unit    – unit tests only (fast)
#   make test-e2e     – end-to-end tests only
#   make lint         – ruff + black --check
#   make format       – auto-format with black + ruff --fix
#   make run          – run the CLI optimizer (quick smoke)
#   make server       – start the HTTP simulator service
#   make guide        – compile website Fleet Sim guide assets
#   make install      – install package in editable mode with dev deps
#   make package      – build sdist + wheel
#   make clean        – remove build/cache artifacts

PYTHON   ?= python3
PIP      ?= $(PYTHON) -m pip
PYTEST   ?= $(PYTHON) -m pytest
BLACK    ?= $(PYTHON) -m black
RUFF     ?= ruff

# Source directories
SRC_DIRS  = fleet_sim
TEST_DIR  = tests
ALL_DIRS  = $(SRC_DIRS) $(TEST_DIR)

# Default CDF and parameters for smoke runs
CDF      ?= data/azure_cdf.json
LAM      ?= 50
SLO      ?= 500
HOST     ?= 0.0.0.0
PORT     ?= 8000
GUIDE_DIR ?= ../../website/static/files/fleet-sim
GUIDE_TEX ?= $(GUIDE_DIR)/fleet-sim.tex
GUIDE_BIB ?= $(GUIDE_DIR)/refs.bib
GUIDE_PDF ?= $(GUIDE_DIR)/fleet-sim.pdf

.PHONY: all test test-unit test-e2e lint format run server guide \
        install install-api install-dev package clean help

all: lint test

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST) $(TEST_DIR) -v

test-unit:
	$(PYTEST) $(TEST_DIR) -v \
	  --ignore=$(TEST_DIR)/test_e2e.py

test-e2e:
	$(PYTEST) $(TEST_DIR)/test_e2e.py -v

test-cov:
	$(PYTEST) $(TEST_DIR) -v \
	  --cov=fleet_sim --cov-report=term-missing --cov-report=html

# ── Lint and format ───────────────────────────────────────────────────────────

lint:
	@echo "── ruff ──────────────────────────────────────────────────────────────"
	$(RUFF) check $(ALL_DIRS) || true
	@echo "── black (check only) ────────────────────────────────────────────────"
	$(BLACK) --check --diff $(ALL_DIRS)

format:
	@echo "── ruff --fix ────────────────────────────────────────────────────────"
	$(RUFF) check --fix $(ALL_DIRS)
	@echo "── black ─────────────────────────────────────────────────────────────"
	$(BLACK) $(ALL_DIRS)

# ── Running ───────────────────────────────────────────────────────────────────

run:
	$(PYTHON) run_sim.py optimize \
	  --cdf $(CDF) --lam $(LAM) --slo $(SLO) \
	  --b-short 4096 --verify-top 1 --n-sim-req 5000

simulate:
	@echo "Usage: make simulate N_S=<n> N_L=<m> GAMMA=<g>"
	$(PYTHON) run_sim.py simulate \
	  --cdf $(CDF) --lam $(LAM) \
	  --n-s $(N_S) --n-l $(N_L) --gamma $(GAMMA) \
	  --b-short 4096 --n-req 10000

whatif:
	$(PYTHON) run_sim.py whatif \
	  --cdf $(CDF) --lam-range 20 50 100 200 \
	  --slo $(SLO) --b-short 4096

compare-routers:
	$(PYTHON) run_sim.py compare-routers \
	  --cdf $(CDF) --lam $(LAM) --n-s 10 --n-l 4 \
	  --b-short 4096 --n-req 10000

server:
	$(PYTHON) run_sim.py serve --host $(HOST) --port $(PORT)

# ── Documentation ─────────────────────────────────────────────────────────────

guide: $(GUIDE_PDF)

$(GUIDE_PDF): $(GUIDE_TEX) $(GUIDE_BIB)
	cd $(GUIDE_DIR) && pdflatex -interaction=nonstopmode fleet-sim.tex
	cd $(GUIDE_DIR) && bibtex fleet-sim
	cd $(GUIDE_DIR) && pdflatex -interaction=nonstopmode fleet-sim.tex
	cd $(GUIDE_DIR) && pdflatex -interaction=nonstopmode fleet-sim.tex

# ── Installation ──────────────────────────────────────────────────────────────

install:
	$(PIP) install -e .

install-api:
	$(PIP) install -e ".[api]"

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest-cov

# ── Packaging ─────────────────────────────────────────────────────────────────

package:
	$(PIP) install --quiet build
	$(PYTHON) -m build

# ── Utilities ────────────────────────────────────────────────────────────────

list-gpus:
	$(PYTHON) -c "from fleet_sim import list_hardware; print('\n'.join(list_hardware()))"

list-models:
	$(PYTHON) -c "from fleet_sim import list_models; print('\n'.join(list_models()))"

profile-info:
	@echo "Usage: make profile-info GPU=h100 MODEL=llama-3.1-70b TP=8"
	$(PYTHON) -c "\
from fleet_sim import get_hardware, get_model; \
from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig; \
hw = get_hardware('$(GPU)'); model = get_model('$(MODEL)'); \
cfg = ServingConfig(tp=$(TP), dtype_bytes=2, mean_ctx_tokens=2048); \
p = ProfileBuilder().build(hw, model, cfg); print(p.summary())"

hf-spec:
	@echo "Usage: make hf-spec HF_MODEL=meta-llama/Meta-Llama-3.1-8B"
	$(PYTHON) -c "\
from fleet_sim.models.spec import ModelSpec; \
s = ModelSpec.from_hf_config('$(HF_MODEL)'); \
print(f'name={s.name}  layers={s.n_layers}  hidden={s.hidden_size}  is_moe={s.is_moe}')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .ruff_cache/ htmlcov/ .coverage
	rm -f $(GUIDE_DIR)/fleet-sim.aux $(GUIDE_DIR)/fleet-sim.bbl $(GUIDE_DIR)/fleet-sim.blg
	rm -f $(GUIDE_DIR)/fleet-sim.log $(GUIDE_DIR)/fleet-sim.out $(GUIDE_DIR)/fleet-sim.toc

help:
	@echo ""
	@echo "vllm-sr-sim — available targets"
	@echo "──────────────────────────────────────────────────────────"
	@echo "  test            Run full test suite"
	@echo "  test-unit       Unit tests only (no e2e)"
	@echo "  test-e2e        End-to-end optimizer + DES simulation tests"
	@echo "  test-cov        Tests with HTML coverage report"
	@echo "  lint            ruff + black --check"
	@echo "  format          Auto-format (ruff --fix + black)"
	@echo "  run             Optimize a fleet (CDF=... LAM=... SLO=...)"
	@echo "  simulate        Simulate a fixed fleet (N_S=... N_L=... GAMMA=...)"
	@echo "  whatif          Sweep arrival rates"
	@echo "  compare-routers Benchmark routing algorithms"
	@echo "  server          Start HTTP simulator service (HOST=... PORT=...)"
	@echo "  guide           Compile website/static/files/fleet-sim/fleet-sim.tex → fleet-sim.pdf"
	@echo "  install         pip install -e ."
	@echo "  install-api     pip install -e .[api]"
	@echo "  install-dev     pip install -e .[dev]"
	@echo "  package         Build sdist + wheel"
	@echo "  list-gpus       Print hardware catalog"
	@echo "  list-models     Print model catalog"
	@echo "  profile-info    GPU=h100 MODEL=llama-3.1-70b TP=8"
	@echo "  hf-spec         HF_MODEL=meta-llama/Meta-Llama-3.1-8B"
	@echo "  clean           Remove build/cache artifacts"
	@echo ""
